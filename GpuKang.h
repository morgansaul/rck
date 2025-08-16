#pragma once

#include "Ec.h"
#include "defs.h"

#define STATS_WND_SIZE    16

struct EcJMP
{
    EcPoint p;
    EcInt dist;
};

// 96 bytes size
struct TPointPriv
{
    u64 x[4];
    u64 y[4];
    u64 priv[4];
};

// [MODIFIED] Extended GPU kangaroo class
class RCGpuKang
{
private:
    bool StopFlag;
    EcPoint* PntsToSolve;  // Changed from single point to array
    int PubKeyCount;       // Number of loaded pubkeys
    int Range;            // in bits
    int DP;               // in bits
    Ec ec;

    u32* DPs_out;
    TKparams Kparams;

    EcInt HalfRange;
    EcPoint PntHalfRange;
    EcPoint NegPntHalfRange;
    TPointPriv* RndPnts;
    EcJMP* EcJumps1;
    EcJMP* EcJumps2;
    EcJMP* EcJumps3;

    EcPoint* PntA;        // Arrays for each pubkey
    EcPoint* PntB;
    
    int cur_stats_ind;
    int SpeedStats[STATS_WND_SIZE];

    void GenerateRndDistances();
    bool Start();
    void Release();
#ifdef DEBUG_MODE
    int Dbg_CheckKangs();
#endif

public:
    int persistingL2CacheMaxSize;
    int CudaIndex; // GPU index in cuda
    int mpCnt;
    int KangCnt;
    bool Failed;
    bool IsOldGpu;

    int CalcKangCnt();
    
    // [MODIFIED] Updated Prepare() for multiple pubkeys
    bool Prepare(EcPoint* _PntsToSolve, int _PubKeyCount, int _Range, int _DP, 
                EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3);
                
    void Stop();
    void Execute();

    u32 dbg[256];

    int GetStatsSpeed();
};

// [ADDED] Helper functions for GPU operations
__device__ bool CheckPointAgainstAllTargets(const EcPoint& point, const TPubKey* targets, int count);
__device__ int FindMatchingPubKey(const EcPoint& point, const TPubKey* targets, int count);
