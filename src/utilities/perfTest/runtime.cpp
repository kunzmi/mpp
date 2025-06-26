#include "runtime.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <vector>

namespace mpp
{
Runtime::Runtime(const std::vector<float> &aTimings, float aTotal) : Total(aTotal)
{
    Min = *std::min_element(aTimings.begin(), aTimings.end());
    Max = *std::max_element(aTimings.begin(), aTimings.end());

    const float sum    = std::accumulate(aTimings.begin(), aTimings.end(), 0.0f);
    const float sumSqr = std::inner_product(aTimings.begin(), aTimings.end(), aTimings.begin(), 0.0f);

    const size_t iterations = aTimings.size();
    const float fiterations = static_cast<float>(aTimings.size());
    if (iterations > 1)
    {
        Std  = std::sqrt((sumSqr - (sum * sum) / fiterations) / (fiterations - 1.0f));
        Mean = sum / fiterations;
    }
    else
    {
        Std  = 0.0f;
        Mean = sum;
    }
}

std::ostream &operator<<(std::ostream &aOs, const Runtime &aRuntime)
{
    aOs << "Total: " << aRuntime.Total << " ms - Mean: " << aRuntime.Mean << " Std: " << aRuntime.Std
        << " Min: " << aRuntime.Min << " Max: " << aRuntime.Max << std::endl;

    return aOs;
}
} // namespace mpp