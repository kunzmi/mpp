#include "runtime.h"
#include "testBase.h"
#include <backends/cuda/event.h>
#include <cstddef>
#include <iostream>

namespace opp
{
std::ostream &operator<<(std::ostream &aOs, const TestResult &aResult)
{
    aOs << aResult.Name << ";";
    aOs << aResult.TotalOPP << ";";
    aOs << aResult.TotalNPP << ";";
    aOs << aResult.MeanOPP << ";";
    aOs << aResult.MeanNPP << ";";
    aOs << aResult.StdOPP << ";";
    aOs << aResult.StdNPP << ";";
    aOs << aResult.MinOPP << ";";
    aOs << aResult.MinNPP << ";";
    aOs << aResult.MaxOPP << ";";
    aOs << aResult.MaxNPP << ";";
    aOs << aResult.AbsoluteDifferenceMSec << ";";
    aOs << aResult.RelativeDifference << std::endl;
    return aOs;
}

TestBase::TestBase(size_t aIterations, size_t aRepeats)
    : iterations(aIterations), repeats(aRepeats), runtimes(aIterations, 0)
{
}

Runtime TestBase::Run()
{
    startGlobal.Record();
    for (size_t i = 0; i < iterations; i++)
    {
        startIter.Record();
        for (size_t r = 0; r < repeats; r++)
        {
            RunOnce();
        }
        endIter.Record();
        endIter.Synchronize();
        runtimes[i] = endIter - startIter;
    }
    endGlobal.Record();
    endGlobal.Synchronize();
    totalRuntime = endGlobal - startGlobal;

    return {runtimes, totalRuntime};
}

void TestBase::WarmUp()
{
    RunOnce();
}

Runtime TestBase::GetRuntime()
{
    return {runtimes, totalRuntime};
}
} // namespace opp