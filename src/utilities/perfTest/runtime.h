#pragma once
#include <iostream>
#include <vector>

namespace opp
{
struct Runtime
{
    float Min{0};
    float Max{0};
    float Mean{0};
    float Std{0};
    float Total{0};

    Runtime() = default;
    Runtime(const std::vector<float> &aTimings, float aTotal);
};

std::ostream &operator<<(std::ostream &aOs, const Runtime &aRuntime);
} // namespace opp