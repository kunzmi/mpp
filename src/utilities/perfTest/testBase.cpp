#include "testBase.h"
#include "runtime.h"
#include <backends/cuda/event.h>
#include <cstddef>
#include <iostream>

namespace mpp
{
std::ostream &operator<<(std::ostream &aOs, const TestResult &aResult)
{
    aOs << aResult.Name << "\t";
    aOs << aResult.TotalMPP << "\t";
    aOs << aResult.TotalNPP << "\t";
    aOs << aResult.MeanMPP << "\t";
    aOs << aResult.MeanNPP << "\t";
    aOs << aResult.StdMPP << "\t";
    aOs << aResult.StdNPP << "\t";
    aOs << aResult.MinMPP << "\t";
    aOs << aResult.MinNPP << "\t";
    aOs << aResult.MaxMPP << "\t";
    aOs << aResult.MaxNPP << "\t";
    aOs << aResult.AbsoluteDifferenceMSec << "\t";
    aOs << aResult.RelativeDifference << std::endl;
    return aOs;
}

bool TestResult::operator<(const TestResult &aOther) const
{
    if (Order1 == aOther.Order1)
    {
        if (Order2 == aOther.Order2)
        {
            return Order3 < aOther.Order3;
        }
        return Order2 < aOther.Order2;
    }
    return Order1 < aOther.Order1;
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
} // namespace mpp