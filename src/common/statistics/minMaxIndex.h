#pragma once
#include <common/defines.h>
#include <common/opp_defs.h>
#include <common/vectorTypes.h>

namespace opp
{

// In order to reduce the number of passed pointers to the Min/Max with Index functions, we define a small structure
// containing the computed indices (one struct per image channel)
struct MinMaxIndex
{
    Vector2<int> IndexMin;
    Vector2<int> IndexMax;
};
struct MinMaxIndexChannel
{
    Vector2<int> IndexMin;
    int ChannelMin;
    Vector2<int> IndexMax;
    int ChannelMax;
};
} // namespace opp
