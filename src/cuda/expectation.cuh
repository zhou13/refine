#ifndef EXPECTATION_CUH
#define EXPECTATION_CUH

#include <vector>
#include "multidim_array.h"

class HostData;
class DeviceData;
class MlOptimiser;
class ExpectationCudaSolver {
public:
    ExpectationCudaSolver(MlOptimiser *ml);
    ~ExpectationCudaSolver();
    void initialize();
    void copyWindowsedImagesToGPU();
    void getShiftedImages();
    void getSquaredDifference();
    void convertSquaredDifferencesToWeights();
    void sumWeights();

private:
    HostData *h;
    DeviceData *d;
    MlOptimiser *ml;
};

#endif
