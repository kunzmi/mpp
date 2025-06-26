#pragma once
#include "runtime.h"
#include <backends/cuda/event.h>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace opp
{

struct TestResult
{
    std::string Name;
    float TotalOPP;
    float TotalNPP;
    float MeanOPP;
    float MeanNPP;
    float StdOPP;
    float StdNPP;
    float MinOPP;
    float MinNPP;
    float MaxOPP;
    float MaxNPP;
    float AbsoluteDifferenceMSec;
    float RelativeDifference;
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
} // namespace opp