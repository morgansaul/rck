#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"
#include "GpuKang.h"

cudaError_t cuSetGpuParams(TKparams Kparams, u64* _jmp2_table);
void CallGpuKernelGen(TKparams Kparams);
void CallGpuKernelABC(TKparams Kparams);
void AddPointsToList(u32* data, int cnt, u64 ops_cnt, int pubkey_id);
extern bool gGenMode;

// Device-side pubkey check
__device__ bool CheckPointAgainstAllTargets(const EcPoint& point, const TPubKey* targets, int count) {
    for(int i = 0; i < count; i++) {
        bool match = true;
        #pragma unroll
        for(int j = 0; j < 4; j++) {
            if(point.x.data[j] != targets[i].x[j]) {
                match = false;
                break;
            }
        }
        if(match) return true;
    }
    return false;
}

// Find matching pubkey index
__device__ int FindMatchingPubKey(const EcPoint& point, const TPubKey* targets, int count) {
    for(int i = 0; i < count; i++) {
        if(memcmp(point.x.data, targets[i].x, 32) == 0) {
            return i;
        }
    }
    return -1;
}

int RCGpuKang::CalcKangCnt() {
    Kparams.BlockCnt = mpCnt;
    Kparams.BlockSize = IsOldGpu ? 512 : 256;
    Kparams.GroupCnt = IsOldGpu ? 64 : 24;
    return Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
}

bool RCGpuKang::Prepare(EcPoint* _PntsToSolve, int _PubKeyCount, int _Range, int _DP, 
                       EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3) {
    PntsToSolve = _PntsToSolve;
    PubKeyCount = _PubKeyCount;
    Range = _Range;
    DP = _DP;
    EcJumps1 = _EcJumps1;
    EcJumps2 = _EcJumps2;
    EcJumps3 = _EcJumps3;
    StopFlag = false;
    Failed = false;
    u64 total_mem = 0;
    memset(dbg, 0, sizeof(dbg));
    memset(SpeedStats, 0, sizeof(SpeedStats));
    cur_stats_ind = 0;

    cudaError_t err;
    err = cudaSetDevice(CudaIndex);
    if (err != cudaSuccess)
        return false;

    Kparams.BlockCnt = mpCnt;
    Kparams.BlockSize = IsOldGpu ? 512 : 256;
    Kparams.GroupCnt = IsOldGpu ? 64 : 24;
    KangCnt = Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
    Kparams.KangCnt = KangCnt;
    Kparams.DP = DP;
    Kparams.KernelA_LDS_Size = 64 * JMP_CNT + 16 * Kparams.BlockSize;
    Kparams.KernelB_LDS_Size = 64 * JMP_CNT;
    Kparams.KernelC_LDS_Size = 96 * JMP_CNT;
    Kparams.IsGenMode = gGenMode;

    // Allocate GPU memory
    if (!IsOldGpu) {
        int L2size = Kparams.KangCnt * (3 * 32);
        total_mem += L2size;
        err = cudaMalloc((void**)&Kparams.L2, L2size);
        if (err != cudaSuccess) {
            printf("GPU %d, Allocate L2 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
            return false;
        }
        size_t size = L2size;
        if (size > persistingL2CacheMaxSize)
            size = persistingL2CacheMaxSize;
        err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);
        
        cudaStreamAttrValue stream_attribute;                                                   
        stream_attribute.accessPolicyWindow.base_ptr = Kparams.L2;
        stream_attribute.accessPolicyWindow.num_bytes = size;										
        stream_attribute.accessPolicyWindow.hitRatio = 1.0;                                     
        stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;             
        stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;  	
        err = cudaStreamSetAttribute(NULL, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
        if (err != cudaSuccess) {
            printf("GPU %d, cudaStreamSetAttribute failed: %s\n", CudaIndex, cudaGetErrorString(err));
            return false;
        }
    }

    size_t size = MAX_DP_CNT * GPU_DP_SIZE + 16;
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.DPs_out, size);
    if (err != cudaSuccess) {
        printf("GPU %d Allocate GpuOut memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    size = KangCnt * 96;
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.Kangs, size);
    if (err != cudaSuccess) {
        printf("GPU %d Allocate pKangs memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    total_mem += JMP_CNT * 96;
    err = cudaMalloc((void**)&Kparams.Jumps1, JMP_CNT * 96);
    if (err != cudaSuccess) {
        printf("GPU %d Allocate Jumps1 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    total_mem += JMP_CNT * 96;
    err = cudaMalloc((void**)&Kparams.Jumps2, JMP_CNT * 96);
    if (err != cudaSuccess) {
        printf("GPU %d Allocate Jumps1 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    total_mem += JMP_CNT * 96;
    err = cudaMalloc((void**)&Kparams.Jumps3, JMP_CNT * 96);
    if (err != cudaSuccess) {
        printf("GPU %d Allocate Jumps3 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    size = 2 * (u64)KangCnt * STEP_CNT;
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.JumpsList, size);
    if (err != cudaSuccess) {
        printf("GPU %d Allocate JumpsList memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    size = (u64)KangCnt * (16 * DPTABLE_MAX_CNT + sizeof(u32));
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.DPTable, size);
    if (err != cudaSuccess) {
        printf("GPU %d Allocate DPTable memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    size = mpCnt * Kparams.BlockSize * sizeof(u64);
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.L1S2, size);
    if (err != cudaSuccess) {
        printf("GPU %d Allocate L1S2 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    size = (u64)KangCnt * MD_LEN * (2 * 32);
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.LastPnts, size);
    if (err != cudaSuccess) {
        printf("GPU %d Allocate LastPnts memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    size = (u64)KangCnt * MD_LEN * sizeof(u64);
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.LoopTable, size);
    if (err != cudaSuccess) {
        printf("GPU %d Allocate LastPnts memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    total_mem += 1024;
    err = cudaMalloc((void**)&Kparams.dbg_buf, 1024);
    if (err != cudaSuccess) {
        printf("GPU %d Allocate dbg_buf memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    size = sizeof(u32) * KangCnt + 8;
    total_mem += size;
    err = cudaMalloc((void**)&Kparams.LoopedKangs, size);
    if (err != cudaSuccess) {
        printf("GPU %d Allocate LoopedKangs memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    // Allocate and copy pubkeys to GPU
    size_t pubkeysSize = PubKeyCount * sizeof(TPubKey);
    err = cudaMalloc((void**)&Kparams.PubKeys, pubkeysSize);
    if (err != cudaSuccess) {
        printf("GPU %d Allocate PubKeys memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }
    
    std::vector<TPubKey> hostPubKeys(PubKeyCount);
    for(int i = 0; i < PubKeyCount; i++) {
        memcpy(hostPubKeys[i].x, PntsToSolve[i].x.data, 32);
        memcpy(hostPubKeys[i].y, PntsToSolve[i].y.data, 32);
    }
    err = cudaMemcpy(Kparams.PubKeys, hostPubKeys.data(), pubkeysSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("GPU %d Copy PubKeys failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    Kparams.PubKeyCount = PubKeyCount;
    DPs_out = (u32*)malloc(MAX_DP_CNT * GPU_DP_SIZE);

    // Copy jump tables
    u64* buf = (u64*)malloc(JMP_CNT * 96);
    for (int i = 0; i < JMP_CNT; i++) {
        memcpy(buf + i * 12, EcJumps1[i].p.x.data, 32);
        memcpy(buf + i * 12 + 4, EcJumps1[i].p.y.data, 32);
        memcpy(buf + i * 12 + 8, EcJumps1[i].dist.data, 32);
    }
    err = cudaMemcpy(Kparams.Jumps1, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("GPU %d, cudaMemcpy Jumps1 failed: %s\n", CudaIndex, cudaGetErrorString(err));
        free(buf);
        return false;
    }

    u64* jmp2_table = (u64*)malloc(JMP_CNT * 64);
    for (int i = 0; i < JMP_CNT; i++) {
        memcpy(buf + i * 12, EcJumps2[i].p.x.data, 32);
        memcpy(jmp2_table + i * 8, EcJumps2[i].p.x.data, 32);
        memcpy(buf + i * 12 + 4, EcJumps2[i].p.y.data, 32);
        memcpy(jmp2_table + i * 8 + 4, EcJumps2[i].p.y.data, 32);
        memcpy(buf + i * 12 + 8, EcJumps2[i].dist.data, 32);
    }
    err = cudaMemcpy(Kparams.Jumps2, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
    free(buf);
    if (err != cudaSuccess) {
        printf("GPU %d, cudaMemcpy Jumps2 failed: %s\n", CudaIndex, cudaGetErrorString(err));
        free(jmp2_table);
        return false;
    }

    err = cuSetGpuParams(Kparams, jmp2_table);
    free(jmp2_table);
    if (err != cudaSuccess) {
        printf("GPU %d, cuSetGpuParams failed: %s!\r\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    buf = (u64*)malloc(JMP_CNT * 96);
    for (int i = 0; i < JMP_CNT; i++) {
        memcpy(buf + i * 12, EcJumps3[i].p.x.data, 32);
        memcpy(buf + i * 12 + 4, EcJumps3[i].p.y.data, 32);
        memcpy(buf + i * 12 + 8, EcJumps3[i].dist.data, 32);
    }
    err = cudaMemcpy(Kparams.Jumps3, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
    free(buf);
    if (err != cudaSuccess) {
        printf("GPU %d, cudaMemcpy Jumps3 failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    printf("GPU %d: allocated %llu MB, %d kangaroos. OldGpuMode: %s\r\n", 
           CudaIndex, total_mem / (1024 * 1024), KangCnt, IsOldGpu ? "Yes" : "No");
    return true;
}

void RCGpuKang::Release() {
    free(RndPnts);
    free(DPs_out);
    cudaFree(Kparams.LoopedKangs);
    cudaFree(Kparams.dbg_buf);
    cudaFree(Kparams.LoopTable);
    cudaFree(Kparams.LastPnts);
    cudaFree(Kparams.L1S2);
    cudaFree(Kparams.DPTable);
    cudaFree(Kparams.JumpsList);
    cudaFree(Kparams.Jumps3);
    cudaFree(Kparams.Jumps2);
    cudaFree(Kparams.Jumps1);
    cudaFree(Kparams.Kangs);
    cudaFree(Kparams.DPs_out);
    if (!IsOldGpu)
        cudaFree(Kparams.L2);
    if (Kparams.PubKeys) 
        cudaFree(Kparams.PubKeys);
    if (PntA) 
        delete[] PntA;
    if (PntB) 
        delete[] PntB;
}

void RCGpuKang::Stop() {
    StopFlag = true;
}

void RCGpuKang::GenerateRndDistances() {
    for (int i = 0; i < KangCnt; i++) {
        EcInt d;
        if (i < KangCnt / 3)
            d.RndBits(Range - 4); // TAME kangs
        else {
            d.RndBits(Range - 1);
            d.data[0] &= 0xFFFFFFFFFFFFFFFE; // must be even
        }
        memcpy(RndPnts[i].priv, d.data, 24);
    }
}

bool RCGpuKang::Start() {
    if (Failed)
        return false;

    cudaError_t err;
    err = cudaSetDevice(CudaIndex);
    if (err != cudaSuccess)
        return false;

    HalfRange.Set(1);
    HalfRange.ShiftLeft(Range - 1);
    PntHalfRange = ec.MultiplyG(HalfRange);
    NegPntHalfRange = PntHalfRange;
    NegPntHalfRange.y.NegModP();

    // Initialize for multiple pubkeys
    PntA = new EcPoint[PubKeyCount];
    PntB = new EcPoint[PubKeyCount];
    for (int i = 0; i < PubKeyCount; i++) {
        PntA[i] = ec.AddPoints(PntsToSolve[i], NegPntHalfRange);
        PntB[i] = PntA[i];
        PntB[i].y.NegModP();
    }

    RndPnts = (TPointPriv*)malloc(KangCnt * 96);
    GenerateRndDistances();

    u8 buf_PntA[64], buf_PntB[64];
    for (int i = 0; i < PubKeyCount; i++) {
        if (i == 0) {
            PntA[i].SaveToBuffer64(buf_PntA);
            PntB[i].SaveToBuffer64(buf_PntB);
        }
    }

    for (int i = 0; i < KangCnt; i++) {
        if (i < KangCnt / 3)
            memset(RndPnts[i].x, 0, 64);
        else if (i < 2 * KangCnt / 3)
            memcpy(RndPnts[i].x, buf_PntA, 64);
        else
            memcpy(RndPnts[i].x, buf_PntB, 64);
    }

    err = cudaMemcpy(Kparams.Kangs, RndPnts, KangCnt * 96, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("GPU %d, cudaMemcpy failed: %s\n", CudaIndex, cudaGetErrorString(err));
        return false;
    }

    CallGpuKernelGen(Kparams);

    err = cudaMemset(Kparams.L1S2, 0, mpCnt * Kparams.BlockSize * 8);
    if (err != cudaSuccess)
        return false;
    cudaMemset(Kparams.dbg_buf, 0, 1024);
    cudaMemset(Kparams.LoopTable, 0, KangCnt * MD_LEN * sizeof(u64));
    return true;
}

#ifdef DEBUG_MODE
int RCGpuKang::Dbg_CheckKangs() {
    int kang_size = mpCnt * Kparams.BlockSize * Kparams.GroupCnt * 96;
    u64* kangs = (u64*)malloc(kang_size);
    cudaError_t err = cudaMemcpy(kangs, Kparams.Kangs, kang_size, cudaMemcpyDeviceToHost);
    int res = 0;
    for (int i = 0; i < KangCnt; i++) {
        EcPoint Pnt, p;
        Pnt.LoadFromBuffer64((u8*)&kangs[i * 12 + 0]);
        EcInt dist;
        dist.Set(0);
        memcpy(dist.data, &kangs[i * 12 + 8], 24);
        bool neg = false;
        if (dist.data[2] >> 63) {
            neg = true;
            memset(((u8*)dist.data) + 24, 0xFF, 16);
            dist.Neg();
        }
        p = ec.MultiplyG_Fast(dist);
        if (neg)
            p.y.NegModP();
        if (i < KangCnt / 3)
            p = p;
        else if (i < 2 * KangCnt / 3)
            p = ec.AddPoints(PntA[0], p); // Using first pubkey for check
        else
            p = ec.AddPoints(PntB[0], p); // Using first pubkey for check
        if (!p.IsEqual(Pnt))
            res++;
    }
    free(kangs);
    return res;
}
#endif

extern u32 gTotalErrors;

void RCGpuKang::Execute() {
    cudaSetDevice(CudaIndex);

    if (!Start()) {
        gTotalErrors++;
        return;
    }

#ifdef DEBUG_MODE
    u64 iter = 1;
#endif
    cudaError_t err;	
    while (!StopFlag) {
        u64 t1 = GetTickCount64();
        cudaMemset(Kparams.DPs_out, 0, 4);
        cudaMemset(Kparams.DPTable, 0, KangCnt * sizeof(u32));
        cudaMemset(Kparams.LoopedKangs, 0, 8);
        CallGpuKernelABC(Kparams);
        
        int cnt;
        err = cudaMemcpy(&cnt, Kparams.DPs_out, 4, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("GPU %d, CallGpuKernel failed: %s\r\n", CudaIndex, cudaGetErrorString(err));
            gTotalErrors++;
            break;
        }
        
        if (cnt >= MAX_DP_CNT) {
            cnt = MAX_DP_CNT;
            printf("GPU %d, gpu DP buffer overflow, some points lost, increase DP value!\r\n", CudaIndex);
        }
        u64 pnt_cnt = (u64)KangCnt * STEP_CNT;

        if (cnt) {
            err = cudaMemcpy(DPs_out, Kparams.DPs_out + 4, cnt * GPU_DP_SIZE, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                gTotalErrors++;
                break;
            }
            for (int i = 0; i < cnt; i++) {
                u32* dp = DPs_out + i * (GPU_DP_SIZE/4);
                int pubkey_id = (dp[9] >> 16) & 0xFFFF; // Extract stored pubkey ID
                AddPointsToList(dp, 1, pnt_cnt, pubkey_id);
            }
        }

        // Debug and stats collection
        cudaMemcpy(dbg, Kparams.dbg_buf, 1024, cudaMemcpyDeviceToHost);
        u32 lcnt;
        cudaMemcpy(&lcnt, Kparams.LoopedKangs, 4, cudaMemcpyDeviceToHost);

        u64 t2 = GetTickCount64();
        u64 tm = t2 - t1;
        if (!tm)
            tm = 1;
        int cur_speed = (int)(pnt_cnt / (tm * 1000));
        SpeedStats[cur_stats_ind] = cur_speed;
        cur_stats_ind = (cur_stats_ind + 1) % STATS_WND_SIZE;

#ifdef DEBUG_MODE
        if ((iter % 300) == 0) {
            int corr_cnt = Dbg_CheckKangs();
            if (corr_cnt) {
                printf("DBG: GPU %d, KANGS CORRUPTED: %d\r\n", CudaIndex, corr_cnt);
                gTotalErrors++;
            }
            else
                printf("DBG: GPU %d, ALL KANGS OK!\r\n", CudaIndex);
        }
        iter++;
#endif
    }

    Release();
}

int RCGpuKang::GetStatsSpeed() {
    int res = SpeedStats[0];
    for (int i = 1; i < STATS_WND_SIZE; i++)
        res += SpeedStats[i];
    return res / STATS_WND_SIZE;
}
