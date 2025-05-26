#pragma once
#include <common/complex.h>
#include <common/defines.h>
#include <common/opp_defs.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>

namespace opp
{

struct HarrisCorner
{
    float K;
    float Scale;

    HarrisCorner(float aK, float aScale) : K(aK), Scale(aScale)
    {
    }

    DEVICE_CODE void operator()(const Vector4<float> &aSrc1, Vector1<float> &aSrcDst) const
    {
        // extract covariance tensor values from input vector:
        // | xx xy |
        // | xy yy |
        //
        const float xx = aSrc1.x;
        const float yy = aSrc1.y;
        const float xy = aSrc1.z;

        const float det    = xx * yy - xy * xy;
        const float trace2 = (xx + yy) * (xx + yy);

        aSrcDst = Scale * (det - K * trace2);
    }
};

} // namespace opp
