#pragma once
#include "GpuKang.h"

class KangarooSolver {
public:
    bool Initialize(const std::vector<EcPoint>& targets, 
                   int range, int dp) {
        if(!gpu.Prepare(targets.data(), targets.size(), range, dp, 
                       jumps1, jumps2, jumps3))
            return false;
        return gpu.Start();
    }

    void Solve() {
        gpu.Execute();
    }

    void GetResults(std::vector<DBRec>& results) {
        results.clear();
        // ... [DP retrieval logic]
    }

private:
    RCGpuKang gpu;
    EcJMP jumps1[JMP_CNT], jumps2[JMP_CNT], jumps3[JMP_CNT];
};
