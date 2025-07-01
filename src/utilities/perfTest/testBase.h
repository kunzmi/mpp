#pragma once
#include "runtime.h"
#include <backends/cuda/event.h>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace mpp
{

struct TestsToRun
{
    bool Arithmetic{true};
    bool DataExchange{true};
    bool Filtering{true};
    bool GeometryTransform{true};
    bool Morphology{true};
    bool Statistics{true};
    bool Threshold{true};
};

struct TestResult
{
    std::string Name;
    float TotalMPP;
    float TotalNPP;
    float MeanMPP;
    float MeanNPP;
    float StdMPP;
    float StdNPP;
    float MinMPP;
    float MinNPP;
    float MaxMPP;
    float MaxNPP;
    float AbsoluteDifferenceMSec;
    float RelativeDifference;
    bool IsAligned;
    int Order1;
    int Order2;
    int Order3;

    [[nodiscard]] bool operator<(const TestResult &aOther) const;
};

std::ostream &operator<<(std::ostream &aOs, const TestResult &aResult);

class TestBase
{
  public:
    TestBase(size_t aIterations, size_t aRepeats);
    virtual ~TestBase() = default;

    TestBase(const TestBase &)     = default;
    TestBase(TestBase &&) noexcept = default;

    TestBase &operator=(const TestBase &)     = default;
    TestBase &operator=(TestBase &&) noexcept = default;

    virtual void Init() = 0;

    Runtime Run();

    void WarmUp();

    Runtime GetRuntime();

  protected:
    virtual void RunOnce() = 0;

  private:
    size_t iterations;
    size_t repeats;
    std::vector<float> runtimes;
    float totalRuntime{0};
    cuda::Event startGlobal;
    cuda::Event startIter;
    cuda::Event endIter;
    cuda::Event endGlobal;
};
} // namespace mpp