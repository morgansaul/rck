// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC

#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h> 
#include <inttypes.h>
#include <stdint.h>
#include <fstream>
#include <algorithm>

#include "cuda_runtime.h"
#include "cuda.h"

#include "defs.h"
#include "utils.h"
#include "GpuKang.h"

#ifndef _WIN32
#include <unistd.h>
#endif

// Constants
#define MAX_PUBKEYS 1024  // Maximum number of pubkeys to process

// Global variables
time_t program_start_time = time(NULL);
EcJMP EcJumps1[JMP_CNT];
EcJMP EcJumps2[JMP_CNT];
EcJMP EcJumps3[JMP_CNT];
RCGpuKang* GpuKangs[MAX_GPU_CNT];
int GpuCnt;
volatile long ThrCnt;
volatile bool gSolved;
EcInt Int_HalfRange;
EcPoint Pnt_HalfRange;
EcPoint Pnt_NegHalfRange;
EcInt Int_TameOffset;
Ec ec;
CriticalSection csAddPoints;
u8* pPntList;
u8* pPntList2;
volatile int PntIndex;
TFastBase db;
std::vector<EcPoint> gPubKeys;  // Changed from single gPntToSolve to vector
EcInt gPrivKey;
volatile u64 TotalOps;
u32 TotalSolved;
u32 gTotalErrors;
u64 PntTotalOps;
bool IsBench;
u32 gDP;
u32 gRange;
EcInt gStart;
bool gStartSet;
char gTamesFileName[1024];
double gMax;
bool gGenMode;
bool gIsOpsLimit;
int gCurrentKeyIndex = 0;  // Track current pubkey being processed
u8 gGPUs_Mask[MAX_GPU_CNT];

#pragma pack(push, 1)
struct DBRec {
    u8 x[12];
    u8 d[22];
    u8 type; //0 - tame, 1 - wild1, 2 - wild2
    u8 pubkey_id; // To track which pubkey this DP belongs to
};
#pragma pack(pop)

void InitGpus() {
    GpuCnt = 0;
    int gcnt = 0;
    cudaGetDeviceCount(&gcnt);
    if (gcnt > MAX_GPU_CNT)
        gcnt = MAX_GPU_CNT;

    if (!gcnt)
        return;

    int drv, rt;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    char drvver[100];
    sprintf(drvver, "%d.%d/%d.%d", drv / 1000, (drv % 100) / 10, rt / 1000, (rt % 100) / 10);

    printf("CUDA devices: %d, CUDA driver/runtime: %s\r\n", gcnt, drvver);
    cudaError_t cudaStatus;
    for (int i = 0; i < gcnt; i++) {
        cudaStatus = cudaSetDevice(i);
        if (cudaStatus != cudaSuccess) {
            printf("cudaSetDevice for gpu %d failed!\r\n", i);
            continue;
        }

        if (!gGPUs_Mask[i])
            continue;

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        printf("GPU %d: %s, %.2f GB, %d CUs, cap %d.%d, L2 size: %d KB\r\n", 
              i, deviceProp.name, 
              ((float)(deviceProp.totalGlobalMem / (1024 * 1024))) / 1024.0f, 
              deviceProp.multiProcessorCount, 
              deviceProp.major, deviceProp.minor, 
              deviceProp.l2CacheSize / 1024);

        if (deviceProp.major < 6) {
            printf("GPU %d - not supported, skip\r\n", i);
            continue;
        }

        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

        GpuKangs[GpuCnt] = new RCGpuKang();
        GpuKangs[GpuCnt]->CudaIndex = i;
        GpuKangs[GpuCnt]->persistingL2CacheMaxSize = deviceProp.persistingL2CacheMaxSize;
        GpuKangs[GpuCnt]->mpCnt = deviceProp.multiProcessorCount;
        GpuKangs[GpuCnt]->IsOldGpu = deviceProp.l2CacheSize < 16 * 1024 * 1024;
        GpuCnt++;
    }
    printf("GPUs Found: %d\r\n", GpuCnt);
}

#ifdef _WIN32
u32 __stdcall kang_thr_proc(void* data) {
#else
void* kang_thr_proc(void* data) {
#endif
    RCGpuKang* Kang = (RCGpuKang*)data;
    Kang->Execute();
    InterlockedDecrement(&ThrCnt);
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

void AddPointsToList(u32* data, int pnt_cnt, u64 ops_cnt, int pubkey_id) {
    csAddPoints.Enter();
    if (PntIndex + pnt_cnt >= MAX_CNT_LIST) {
        csAddPoints.Leave();
        printf("DPs buffer overflow, some points lost, increase DP value!\r\n");
        return;
    }
    
    // Store pubkey ID in the data
    for (int i = 0; i < pnt_cnt; i++) {
        u32* p = data + i * (GPU_DP_SIZE/4);
        p[10] = (p[10] & ~PKID_MASK) | (pubkey_id & PKID_MASK);
    }
    
    memcpy(pPntList + GPU_DP_SIZE * PntIndex, data, pnt_cnt * GPU_DP_SIZE);
    PntIndex += pnt_cnt;
    PntTotalOps += ops_cnt;
    csAddPoints.Leave();
}

bool Collision_SOTA(EcPoint& pnt, EcInt t, int TameType, EcInt w, int WildType, bool IsNeg, int pubkey_index) {
    EcPoint target = gPubKeys[pubkey_index];  // Use specific pubkey
    
    if (IsNeg)
        t.Neg();
    if (TameType == TAME) {
        gPrivKey = t;
        gPrivKey.Sub(w);
        EcInt sv = gPrivKey;
        gPrivKey.Add(Int_HalfRange);
        EcPoint P = ec.MultiplyG(gPrivKey);
        if (P.IsEqual(target))
            return true;
        gPrivKey = sv;
        gPrivKey.Neg();
        gPrivKey.Add(Int_HalfRange);
        P = ec.MultiplyG(gPrivKey);
        return P.IsEqual(target);
    }
    else {
        gPrivKey = t;
        gPrivKey.Sub(w);
        if (gPrivKey.data[4] >> 63)
            gPrivKey.Neg();
        gPrivKey.ShiftRight(1);
        EcInt sv = gPrivKey;
        gPrivKey.Add(Int_HalfRange);
        EcPoint P = ec.MultiplyG(gPrivKey);
        if (P.IsEqual(target))
            return true;
        gPrivKey = sv;
        gPrivKey.Neg();
        gPrivKey.Add(Int_HalfRange);
        P = ec.MultiplyG(gPrivKey);
        return P.IsEqual(target);
    }
}

void trim_leading_zeros(char* str) {
    char* non_zero = str;
    while (*non_zero == '0' && *(non_zero + 1) != '\0') {
        non_zero++;
    }
    if (non_zero != str) {
        memmove(str, non_zero, strlen(non_zero) + 1);
    }
}

void CheckNewPoints() {
    csAddPoints.Enter();
    if (!PntIndex) {
        csAddPoints.Leave();
        return;
    }

    int cnt = PntIndex;
    memcpy(pPntList2, pPntList, GPU_DP_SIZE * cnt);
    PntIndex = 0;
    csAddPoints.Leave();

    for (int i = 0; i < cnt; i++) {
        DBRec nrec;
        u8* p = pPntList2 + i * GPU_DP_SIZE;
        memcpy(nrec.x, p, 12);
        memcpy(nrec.d, p + 16, 22);
        nrec.type = gGenMode ? TAME : p[40];
        nrec.pubkey_id = (p[40] >> 2) & 0x3F;  // Extract pubkey ID from bits 2-7

        DBRec* pref = (DBRec*)db.FindOrAddDataBlock((u8*)&nrec);
        if (gGenMode)
            continue;
        if (pref) {
            // In db we don't store first 3 bytes so restore them
            DBRec tmp_pref;
            memcpy(&tmp_pref, &nrec, 3);
            memcpy(((u8*)&tmp_pref) + 3, pref, sizeof(DBRec) - 3);
            pref = &tmp_pref;

            if (pref->type == nrec.type && pref->pubkey_id == nrec.pubkey_id) {
                if (pref->type == TAME)
                    continue;

                // If it's wild, we can find the key from the same type if distances are different
                if (*(u64*)pref->d == *(u64*)nrec.d)
                    continue;
            }

            EcInt w, t;
            int TameType, WildType;
            if (pref->type != TAME) {
                memcpy(w.data, pref->d, sizeof(pref->d));
                if (pref->d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
                memcpy(t.data, nrec.d, sizeof(nrec.d));
                if (nrec.d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
                TameType = nrec.type;
                WildType = pref->type;
            }
            else {
                memcpy(w.data, nrec.d, sizeof(nrec.d));
                if (nrec.d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
                memcpy(t.data, pref->d, sizeof(pref->d));
                if (pref->d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
                TameType = TAME;
                WildType = nrec.type;
            }

            bool res = Collision_SOTA(gPubKeys[nrec.pubkey_id], t, TameType, w, WildType, false, nrec.pubkey_id) || 
                      Collision_SOTA(gPubKeys[nrec.pubkey_id], t, TameType, w, WildType, true, nrec.pubkey_id);
            if (!res) {
                bool w12 = ((pref->type == WILD1) && (nrec.type == WILD2)) || ((pref->type == WILD2) && (nrec.type == WILD1));
                if (w12) // In rare cases WILD and WILD2 can collide in mirror
                    continue;
                else {
                    printf("Collision Error for pubkey %d\r\n", nrec.pubkey_id);
                    gTotalErrors++;
                }
                continue;
            }
            gSolved = true;
            gCurrentKeyIndex = nrec.pubkey_id;
            break;
        }
    }
}

void ShowStats(u64 tm_start, double exp_ops, double dp_val) {
#ifdef DEBUG_MODE
    for (int i = 0; i <= MD_LEN; i++) {
        u64 val = 0;
        for (int j = 0; j < GpuCnt; j++) {
            val += GpuKangs[j]->dbg[i];
        }
        if (val)
            printf("Loop size %d: %llu\r\n", i, val);
    }
#endif

    int speed = GpuKangs[0]->GetStatsSpeed();
    for (int i = 1; i < GpuCnt; i++)
        speed += GpuKangs[i]->GetStatsSpeed();

    u64 est_dps_cnt = (u64)(exp_ops / dp_val);
    u64 exp_sec = 0xFFFFFFFFFFFFFFFFull;

    if (speed)
        exp_sec = (u64)((exp_ops / 1000000) / speed);  // Expected time in seconds

    // Expected Time Breakdown
    u64 exp_days = exp_sec / (3600 * 24);
    int exp_hours = (int)(exp_sec % (3600 * 24)) / 3600;
    int exp_min = (int)(exp_sec % 3600) / 60;
    int exp_remaining_sec = (int)(exp_sec % 60);

    // Elapsed Time Calculation
    u64 sec = (GetTickCount64() - tm_start) / 1000;
    u64 days = sec / (3600 * 24);
    int hours = (int)(sec % (3600 * 24)) / 3600;
    int min = (int)(sec % 3600) / 60;
    int remaining_sec = (int)(sec % 60);

    printf("[%d/%d] Speed: %d MKeys/s | DPs: %lluK | Time: %llud:%02dh:%02dm:%02ds\r",
          gCurrentKeyIndex + 1, (int)gPubKeys.size(),
          speed,
          db.GetBlockCnt() / 1000,
          days, hours, min, remaining_sec);

    fflush(stdout);
}

bool LoadPubKeysFromFile(const char* path) {
    std::ifstream f(path);
    if (!f) {
        printf("Failed to open pubkeys file: %s\n", path);
        return false;
    }

    gPubKeys.clear();
    EcPoint p;
    std::string line;
    while (std::getline(f, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        // Remove any whitespace
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());

        if (p.SetHexStr(line.c_str())) {
            gPubKeys.push_back(p);
            if (gPubKeys.size() >= MAX_PUBKEYS) {
                printf("Warning: Reached maximum pubkeys limit (%d)\n", MAX_PUBKEYS);
                break;
            }
        } else {
            printf("Warning: Invalid pubkey format: %s\n", line.c_str());
        }
    }

    if (gPubKeys.empty()) {
        printf("No valid pubkeys found in file: %s\n", path);
        return false;
    }

    printf("Loaded %d pubkeys from %s\n", (int)gPubKeys.size(), path);
    return true;
}

bool SolvePoints(int Range, int DP, EcInt* pk_res) {
    if ((Range < 32) || (Range > 180)) {
        printf("Unsupported Range value (%d)!\r\n", Range);
        return false;
    }
    if ((DP < 14) || (DP > 60)) {
        printf("Unsupported DP value (%d)!\r\n", DP);
        return false;
    }

    printf("\r\nSolving %d pubkeys: Range %d bits, DP %d, start...\r\n", (int)gPubKeys.size(), Range, DP);
    double ops = 1.15 * pow(2.0, Range / 2.0);
    double dp_val = (double)(1ull << DP);
    double ram = (32 + 4 + 4) * ops / dp_val;
    ram += sizeof(TListRec) * 256 * 256 * 256;
    ram /= (1024 * 1024 * 1024);
    printf("SOTA method, estimated ops: 2^%.3f, RAM for DPs: %.3f GB. DP and GPU overheads not included!\r\n", log2(ops), ram);
    gIsOpsLimit = false;
    double MaxTotalOps = 0.0;
    if (gMax > 0) {
        MaxTotalOps = gMax * ops;
        double ram_max = (32 + 4 + 4) * MaxTotalOps / dp_val;
        ram_max += sizeof(TListRec) * 256 * 256 * 256;
        ram_max /= (1024 * 1024 * 1024);
        printf("Max allowed number of ops: 2^%.3f, max RAM for DPs: %.3f GB\r\n", log2(MaxTotalOps), ram_max);
    }

    u64 total_kangs = GpuKangs[0]->CalcKangCnt();
    for (int i = 1; i < GpuCnt; i++)
        total_kangs += GpuKangs[i]->CalcKangCnt();
    double path_single_kang = ops / total_kangs;
    double DPs_per_kang = path_single_kang / dp_val;
    printf("Estimated DPs per kangaroo: %.3f.%s\r\n", DPs_per_kang, (DPs_per_kang < 5) ? " DP overhead is big, use less DP value if possible!" : "");

    if (!gGenMode && gTamesFileName[0]) {
        printf("load tames...\r\n");
        if (db.LoadFromFile(gTamesFileName)) {
            printf("tames loaded\r\n");
            if (db.Header[0] != gRange) {
                printf("loaded tames have different range, they cannot be used, clear\r\n");
                db.Clear();
            }
        }
        else
            printf("tames loading failed\r\n");
    }

    SetRndSeed(0);
    PntTotalOps = 0;
    PntIndex = 0;

    // Prepare jumps
    EcInt minjump, t;
    minjump.Set(1);
    minjump.ShiftLeft(Range / 2 + 3);
    for (int i = 0; i < JMP_CNT; i++) {
        EcJumps1[i].dist = minjump;
        t.RndMax(minjump);
        EcJumps1[i].dist.Add(t);
        EcJumps1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        EcJumps1[i].p = ec.MultiplyG(EcJumps1[i].dist);
    }

    minjump.Set(1);
    minjump.ShiftLeft(Range - 10);
    for (int i = 0; i < JMP_CNT; i++) {
        EcJumps2[i].dist = minjump;
        t.RndMax(minjump);
        EcJumps2[i].dist.Add(t);
        EcJumps2[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        EcJumps2[i].p = ec.MultiplyG(EcJumps2[i].dist);
    }

    minjump.Set(1);
    minjump.ShiftLeft(Range - 10 - 2);
    for (int i = 0; i < JMP_CNT; i++) {
        EcJumps3[i].dist = minjump;
        t.RndMax(minjump);
        EcJumps3[i].dist.Add(t);
        EcJumps3[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        EcJumps3[i].p = ec.MultiplyG(EcJumps3[i].dist);
    }
    SetRndSeed(GetTickCount64());

    Int_HalfRange.Set(1);
    Int_HalfRange.ShiftLeft(Range - 1);
    Pnt_HalfRange = ec.MultiplyG(Int_HalfRange);
    Pnt_NegHalfRange = Pnt_HalfRange;
    Pnt_NegHalfRange.y.NegModP();
    Int_TameOffset.Set(1);
    Int_TameOffset.ShiftLeft(Range - 1);
    EcInt tt;
    tt.Set(1);
    tt.ShiftLeft(Range - 5);
    Int_TameOffset.Sub(tt);

    // Prepare GPUs for all pubkeys
    for (int i = 0; i < GpuCnt; i++) {
        if (!GpuKangs[i]->Prepare(gPubKeys.data(), (int)gPubKeys.size(), Range, DP, EcJumps1, EcJumps2, EcJumps3)) {
            printf("GPU %d Prepare failed\r\n", GpuKangs[i]->CudaIndex);
            GpuKangs[i]->Failed = true;
        }
    }

    u64 tm0 = GetTickCount64();
    printf("GPUs started...\r\n");

#ifdef _WIN32
    HANDLE thr_handles[MAX_GPU_CNT];
#else
    pthread_t thr_handles[MAX_GPU_CNT];
#endif

    u32 ThreadID;
    gSolved = false;
    ThrCnt = GpuCnt;
    for (int i = 0; i < GpuCnt; i++) {
#ifdef _WIN32
        thr_handles[i] = (HANDLE)_beginthreadex(NULL, 0, kang_thr_proc, (void*)GpuKangs[i], 0, &ThreadID);
#else
        pthread_create(&thr_handles[i], NULL, kang_thr_proc, (void*)GpuKangs[i]);
#endif
    }

    u64 tm_stats = GetTickCount64();
    while (!gSolved && !gIsOpsLimit) {
        CheckNewPoints();
    
    #ifdef _WIN32
        Sleep(5);
    #else
        usleep(5000);
    #endif
    
        if (GetTickCount64() - tm_stats > 10000) {
            printf("Keys checked: %d/%d\n", gCurrentKeyIndex + 1, (int)gPubKeys.size());
            ShowStats(tm0, ops, dp_val);
            tm_stats = GetTickCount64();
        }
    }

    printf("\n\nStopping work ...\r\n");
    time_t program_end_time = time(NULL);
    time_t total_seconds = program_end_time - program_start_time;

    int total_days = total_seconds / (24 * 3600);
    int total_hours = (total_seconds % (24 * 3600)) / 3600;
    int total_minutes = (total_seconds % 3600) / 60;
    int remaining_seconds = total_seconds % 60;

    printf("Total Time: ");
    int printed = 0;
    if (total_days > 0) {
        printf("%d day%s", total_days, total_days == 1 ? "" : "s");
        printed = 1;
    }
    if (total_hours > 0) {
        if (printed) printf(", ");
        printf("%d hour%s", total_hours, total_hours == 1 ? "" : "s");
        printed = 1;
    }
    if (total_minutes > 0) {
        if (printed) printf(", ");
        printf("%d minute%s", total_minutes, total_minutes == 1 ? "" : "s");
        printed = 1;
    }
    if (remaining_seconds > 0 || !printed) {
        if (printed) printf(", ");
        printf("%d second%s", remaining_seconds, remaining_seconds == 1 ? "" : "s");
    }
    printf("\n");

    for (int i = 0; i < GpuCnt; i++)
        GpuKangs[i]->Stop();
    while (ThrCnt)
        Sleep(10);
    for (int i = 0; i < GpuCnt; i++) {
#ifdef _WIN32
        CloseHandle(thr_handles[i]);
#else
        pthread_join(thr_handles[i], NULL);
#endif
    }

    if (gIsOpsLimit) {
        if (gGenMode) {
            printf("saving tames...\r\n");
            db.Header[0] = gRange;
            if (db.SaveToFile(gTamesFileName))
                printf("tames saved\r\n");
            else
                printf("tames saving failed\r\n");
        }
        db.Clear();
        return false;
    }

    double K = (double)PntTotalOps / pow(2.0, Range / 2.0);
    printf("Point solved, K: %.3f (with DP and GPU overheads)\r\n\r\n", K);
    db.Clear();
    *pk_res = gPrivKey;
    return true;
}

bool ParseCommandLine(int argc, char* argv[]) {
    int ci = 1;
    while (ci < argc) {
        char* argument = argv[ci];
        ci++;

        if (strcmp(argument, "-gpu") == 0) {
            if (ci >= argc) {
                printf("error: missed value after -gpu option\r\n");
                return false;
            }
            char* gpus = argv[ci++];
            memset(gGPUs_Mask, 0, sizeof(gGPUs_Mask));
            for (int i = 0; i < (int)strlen(gpus); i++) {
                if ((gpus[i] < '0') || (gpus[i] > '9')) {
                    printf("error: invalid value for -gpu option\r\n");
                    return false;
                }
                gGPUs_Mask[gpus[i] - '0'] = 1;
            }
        }
        else if (strcmp(argument, "-dp") == 0) {
            if (ci >= argc) {
                printf("error: missed value after -dp option\r\n");
                return false;
            }
            int val = atoi(argv[ci++]);
            if (val < 14 || val > 60) {
                printf("error: invalid value for -dp option\r\n");
                return false;
            }
            gDP = val;
        }
        else if (strcmp(argument, "-range") == 0) {
            char* range_str = argv[ci];
            ci++;

            char* colon_pos = strchr(range_str, ':');
            if (colon_pos == NULL) {
                printf("error: invalid format for -range option, expected start:end in hex\n");
                return false;
            }

            *colon_pos = '\0';
            char* start_str = range_str;
            char* end_str = colon_pos + 1;

            if (!gStart.SetHexStr(start_str)) {
                printf("error: failed to set gStart from range start value\n");
                return false;
            }
            EcInt gEnd;
            if (!gEnd.SetHexStr(end_str)) {
                printf("error: failed to set gEnd from range end value\n");
                return false;
            }

            uint64_t* gStartWords = (uint64_t*)gStart.data;
            if (gStartWords[0] == 0 && gStartWords[1] == 0 &&
                gStartWords[2] == 0 && gStartWords[3] == 0) {
                printf("error: gStart is zero after assignment — check input!\n");
                return false;
            }

            char start_range_str[100], end_range_str[100];
            gStart.GetHexStr(start_range_str);
            gEnd.GetHexStr(end_range_str);
            trim_leading_zeros(start_range_str);
            trim_leading_zeros(end_range_str);
            printf("Start Range: %s\n", start_range_str);
            printf("End   Range: %s\n", end_range_str);

            EcInt rangeDiff(gEnd);
            if (rangeDiff.Sub(gStart)) {
                printf("error: end value must be greater than start value\n");
                return false;
            }

            int bitlen = 0;
            for (int i = 31; i >= 0; --i) {
                uint8_t byte = ((uint8_t*)rangeDiff.data)[i];
                if (byte) {
                    for (int b = 7; b >= 0; --b) {
                        if (byte & (1 << b)) {
                            bitlen = i * 8 + b + 1;
                            break;
                        }
                    }
                    break;
                }
            }

            gRange = bitlen;
            printf("Bits: %d\n", gRange);

            if (gRange < 32 || gRange > 180) {
                printf("error: invalid range, resulting bit length must be between 32 and 180\n");
                return false;
            }
        }
        else if (strcmp(argument, "-pubkey") == 0) {
            if (ci >= argc) {
                printf("error: missed value after -pubkey option\r\n");
                return false;
            }
            EcPoint p;
            if (!p.SetHexStr(argv[ci++])) {
                printf("error: invalid value for -pubkey option\r\n");
                return false;
            }
            gPubKeys.push_back(p);
        }
        else if (strcmp(argument, "-pubkeys") == 0) {
            if (ci >= argc) {
                printf("error: missed value after -pubkeys option\r\n");
                return false;
            }
            if (!LoadPubKeysFromFile(argv[ci++])) {
                printf("error: failed to load pubkeys from file\r\n");
                return false;
            }
        }
        else if (strcmp(argument, "-tames") == 0) {
            if (ci >= argc) {
                printf("error: missed value after -tames option\r\n");
                return false;
            }
            strcpy(gTamesFileName, argv[ci++]);
        }
        else if (strcmp(argument, "-max") == 0) {
            if (ci >= argc) {
                printf("error: missed value after -max option\r\n");
                return false;
            }
            double val = atof(argv[ci++]);
            if (val < 0.001) {
                printf("error: invalid value for -max option\r\n");
                return false;
            }
            gMax = val;
        }
        else {
            printf("error: unknown option %s\r\n", argument);
            return false;
        }
    }

    if (gPubKeys.empty() && !gGenMode) {
        printf("error: no pubkeys specified\r\n");
        return false;
    }

    if (gTamesFileName[0] && !IsFileExist(gTamesFileName)) {
        if (gMax == 0.0) {
            printf("error: you must also specify -max option to generate tames\r\n");
            return false;
        }
        gGenMode = true;
    }

    return true;
}

int main(int argc, char* argv[]) {
#ifdef _DEBUG    
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

    printf("********************************************************************************\r\n");
    printf("*                    RCKangaroo v3.0  (c) 2024 RetiredCoder                    *\r\n");
    printf("********************************************************************************\r\n\r\n");

    printf("This software is free and open-source: https://github.com/RetiredC\r\n");
    printf("It demonstrates fast GPU implementation of SOTA Kangaroo method for solving ECDLP\r\n");

#ifdef _WIN32
    printf("Windows version\r\n");
#else
    printf("Linux version\r\n");
#endif

#ifdef DEBUG_MODE
    printf("DEBUG MODE\r\n\r\n");
#endif

    InitEc();
    gDP = 0;
    gRange = 0;
    gStartSet = false;
    gTamesFileName[0] = 0;
    gMax = 0.0;
    gGenMode = false;
    gIsOpsLimit = false;
    memset(gGPUs_Mask, 1, sizeof(gGPUs_Mask));
    if (!ParseCommandLine(argc, argv))
        return 0;

    InitGpus();

    if (!GpuCnt) {
        printf("No supported GPUs detected, exit\r\n");
        return 0;
    }

    pPntList = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
    pPntList2 = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
    TotalOps = 0;
    TotalSolved = 0;
    gTotalErrors = 0;
    IsBench = gPubKeys.empty() && !gGenMode;

    if (!IsBench && !gGenMode) {
        printf("\r\nMAIN MODE\r\n\r\n");
        EcInt pk_found;

        if (!gStart.IsZero()) {
            for (size_t i = 0; i < gPubKeys.size(); i++) {
                EcPoint PntOfs = ec.MultiplyG(gStart);
                PntOfs.y.NegModP();
                gPubKeys[i] = ec.AddPoints(gPubKeys[i], PntOfs);
            }
        }

        for (size_t i = 0; i < gPubKeys.size(); i++) {
            char sx[100], sy[100];
            gPubKeys[i].x.GetHexStr(sx);
            gPubKeys[i].y.GetHexStr(sy);
            printf("Solving pubkey %d/%d\r\nX: %s\r\nY: %s\r\n", 
                  (int)i+1, (int)gPubKeys.size(), sx, sy);
        }
        
        if (!gStart.IsZero()) {
            char sx[100];
            gStart.GetHexStr(sx);
            trim_leading_zeros(sx);
            printf("Offset: %s\n", sx);
        }

        if (!SolvePoints(gRange, gDP, &pk_found)) {
            if (!gIsOpsLimit)
                printf("FATAL ERROR: SolvePoints failed\r\n");
            goto label_end;
        }
        
        pk_found.AddModP(gStart);
        EcPoint tmp = ec.MultiplyG(pk_found);
        if (!tmp.IsEqual(gPubKeys[gCurrentKeyIndex])) {
            printf("FATAL ERROR: SolvePoints found incorrect key\r\n");
            goto label_end;
        }

        char s[100];
        pk_found.GetHexStr(s);
        trim_leading_zeros(s);
        printf("\r\nPRIVATE KEY FOUND FOR PUBKEY %d: %s\r\n\r\n", gCurrentKeyIndex+1, s);
        FILE* fp = fopen("RESULTS.TXT", "a");
        if (fp) {
            fprintf(fp, "PUBKEY %d PRIVATE KEY: %s\n", gCurrentKeyIndex+1, s);
            fclose(fp);
        }
        else {
            printf("WARNING: Cannot save the key to RESULTS.TXT!\r\n");
            while (1)
                Sleep(100);
        }
    }
    else {
        if (gGenMode)
            printf("\r\nTAMES GENERATION MODE\r\n");
        else
            printf("\r\nBENCHMARK MODE\r\n");
        
        while (1) {
            EcInt pk, pk_found;
            EcPoint PntToSolve;

            if (!gRange)
                gRange = 78;
            if (!gDP)
                gDP = 16;

            pk.RndBits(gRange);
            PntToSolve = ec.MultiplyG(pk);

            if (!SolvePoints(gRange, gDP, &pk_found)) {
                if (!gIsOpsLimit)
                    printf("FATAL ERROR: SolvePoints failed\r\n");
                break;
            }
            if (!pk_found.IsEqual(pk)) {
                printf("FATAL ERROR: Found key is wrong!\r\n");
                break;
            }

            TotalOps += PntTotalOps;
            TotalSolved++;
            u64 ops_per_pnt = TotalOps / TotalSolved;
            double K = (double)ops_per_pnt / pow(2.0, gRange / 2.0);
            printf("Points solved: %d, average K: %.3f (with DP and GPU overheads)\r\n", TotalSolved, K);
        }
    }

label_end:
    for (int i = 0; i < GpuCnt; i++)
        delete GpuKangs[i];
    DeInitEc();
    free(pPntList2);
    free(pPntList);
    return 0;
}

