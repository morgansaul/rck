#include "RCGpuUtils.h"
#include "defs.h"

__constant__ u64 c_P[4] = {0xFFFFFFFEFFFFFC2Full, 0xFFFFFFFFFFFFFFFFull,
                          0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull};

// Device functions for multi-pubkey support
__device__ int FindMatchingPubKey(const EcPoint& point, const TPubKey* targets, int count) {
    for(int i = 0; i < count; i++) {
        bool match = true;
        #pragma unroll
        for(int j = 0; j < 4; j++) {
            if(point.x.data[j] != targets[i].x[j]) {
                match = false;
                break;
            }
        }
        if(match) return i;
    }
    return 0xFFFF; // Invalid ID
}

__device__ void ProcessDP(TKparams params, const TPointPriv& kang, const EcPoint& pnt) {
    int dp_index = atomicAdd(params.DPs_out, 1);
    if(dp_index >= MAX_DP_CNT) return;

    u32* out = params.DPs_out + 4 + dp_index * (GPU_DP_SIZE/4);
    
    // Store first 12 bytes of X coordinate
    #pragma unroll
    for(int i = 0; i < 3; i++)
        out[i] = ((u32*)kang.x)[i];
    
    // Store 22 bytes of private key
    #pragma unroll
    for(int i = 0; i < 5; i++)
        out[4 + i] = ((u32*)kang.priv)[i];
    
    // Store kangaroo type and pubkey ID
    int pkid = FindMatchingPubKey(pnt, params.PubKeys, params.PubKeyCount);
    out[9] = (pkid << 16) | (params.IsGenMode ? TAME : 
             (kang.priv[0] & 1) ? WILD1 : WILD2) | DP_FLAG;
}

__device__ void ProcessJump(TKparams params, EcPoint& pnt, EcInt& dist, int jump_index) {
    u64* jump = (jump_index & JMP2_FLAG) ? 
                params.Jumps2 + (jump_index & JMP_MASK) * 12 :
                params.Jumps1 + (jump_index & JMP_MASK) * 12;
    
    EcPoint jmp_pnt;
    jmp_pnt.x.data[0] = jump[0];
    jmp_pnt.x.data[1] = jump[1];
    jmp_pnt.x.data[2] = jump[2];
    jmp_pnt.x.data[3] = jump[3];
    jmp_pnt.y.data[0] = jump[4];
    jmp_pnt.y.data[1] = jump[5];
    jmp_pnt.y.data[2] = jump[6];
    jmp_pnt.y.data[3] = jump[7];
    
    EcInt jmp_dist;
    jmp_dist.data[0] = jump[8];
    jmp_dist.data[1] = jump[9];
    jmp_dist.data[2] = jump[10];
    jmp_dist.data[3] = jump[11];
    
    if(jump_index & INV_FLAG) {
        jmp_pnt.y.NegModP();
        jmp_dist.Neg();
    }
    
    AddModP(pnt.x.data, pnt.x.data, jmp_pnt.x.data);
    AddModP(pnt.y.data, pnt.y.data, jmp_pnt.y.data);
    dist.Add(jmp_dist);
}

__global__ void KangarooKernel(TKparams params) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= params.KangCnt) return;

    // Load kangaroo state
    TPointPriv kang;
    *((int4*)&kang.x[0]) = *((int4*)(params.Kangs + gid * 12));
    *((int4*)&kang.y[0]) = *((int4*)(params.Kangs + gid * 12 + 4));
    *((int4*)&kang.priv[0]) = *((int4*)(params.Kangs + gid * 12 + 8));

    EcPoint pnt;
    memcpy(pnt.x.data, kang.x, 32);
    memcpy(pnt.y.data, kang.y, 32);
    EcInt dist;
    memcpy(dist.data, kang.priv, 32);

    // Main kangaroo loop
    for(int step = 0; step < STEP_CNT; step++) {
        // Check for DP
        if((pnt.x.data[0] & ((1ull << params.DP) - 1)) == 0) {
            ProcessDP(params, kang, pnt);
        }

        // Calculate jump index
        u32 hash = (pnt.x.data[0] >> params.DP) & JMP_MASK;
        int jump_index = (hash >> 10) | ((hash & 0x200) ? JMP2_FLAG : 0);
        
        // Apply jump
        ProcessJump(params, pnt, dist, jump_index);

        // Store state for next iteration
        memcpy(kang.x, pnt.x.data, 32);
        memcpy(kang.y, pnt.y.data, 32);
        memcpy(kang.priv, dist.data, 32);
    }

    // Save final state
    *((int4*)(params.Kangs + gid * 12)) = *((int4*)&kang.x[0]);
    *((int4*)(params.Kangs + gid * 12 + 4)) = *((int4*)&kang.y[0]);
    *((int4*)(params.Kangs + gid * 12 + 8)) = *((int4*)&kang.priv[0]);
}

void CallGpuKernelABC(TKparams params) {
    dim3 blocks(params.BlockCnt);
    dim3 threads(params.BlockSize);
    KangarooKernel<<<blocks, threads, params.KernelA_LDS_Size>>>(params);
    cudaDeviceSynchronize();
}

void CallGpuKernelGen(TKparams params) {
    dim3 blocks(params.BlockCnt);
    dim3 threads(params.BlockSize);
    KangarooGenKernel<<<blocks, threads>>>(params);
    cudaDeviceSynchronize();
}

__global__ void KangarooGenKernel(TKparams params) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= params.KangCnt) return;

    TPointPriv kang;
    if(gid < params.KangCnt / 3) {
        // Tame kangaroos
        memset(kang.x, 0, 32);
    } else {
        // Wild kangaroos
        memcpy(kang.x, params.PubKeys[0].x, 32); // Use first pubkey as base
        if(gid >= 2 * params.KangCnt / 3) {
            // Second wild group uses negated base
            NegModP(kang.x);
        }
    }

    // Initialize random distances
    for(int i = 0; i < 3; i++)
        kang.priv[i] = (u64)blockIdx.x * 0x123456789ABCDEFull + 
                      (u64)threadIdx.x * 0xFEDCBA987654321ull + 
                      (u64)i * 0x13579BDF02468ACEull;
    kang.priv[3] = 0;

    // Save initialized state
    *((int4*)(params.Kangs + gid * 12)) = *((int4*)&kang.x[0]);
    *((int4*)(params.Kangs + gid * 12 + 4)) = *((int4*)&kang.y[0]);
    *((int4*)(params.Kangs + gid * 12 + 8)) = *((int4*)&kang.priv[0]);
}
