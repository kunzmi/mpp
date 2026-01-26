#pragma once
#include <common/defines.h>
#include <common/mpp_defs.h>
#include <common/vectorTypes.h>

namespace mpp
{

// In order to reduce the number of passed pointers to the Min/Max with Index functions, we define a small structure
// containing the computed indices (one struct per image channel)
struct IndexMinMax
{
    Vector2<int> IndexMin;
    Vector2<int> IndexMax;
};
struct IndexMinMaxChannel
{
    Vector2<int> IndexMin;
    Vector2<int> IndexMax;
    int ChannelMin;
    int ChannelMax;
};
} // namespace mpp
