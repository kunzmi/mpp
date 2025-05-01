#pragma once
#include <common/defines.h>
#include <common/image/affineTransformation.h>
#include <common/image/gotoPtr.h>
#include <common/image/matrix.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/size2D.h>
#include <common/opp_defs.h>
#include <common/utilities.h>
#include <common/vector2.h>
#include <concepts>

#ifdef IS_HOST_COMPILER
#include <utility>
#define STD std::
#else
#define STD
#endif
#include <numbers>

namespace opp::image
{
template <typename CoordT, MirrorAxis mirrorAxis> struct TransformerMirror
{
    const Size2D roiSize;

#pragma region Constructors
    TransformerMirror(const Size2D &aRoiSize2D) : roiSize(aRoiSize2D)
    {
    }
#pragma endregion

    DEVICE_CODE Vector2<CoordT> operator()(int aPixelX, int aPixelY) const
    {
        if constexpr (mirrorAxis == MirrorAxis::Horizontal)
        {
            CoordT y = static_cast<CoordT>(roiSize.y) - static_cast<CoordT>(aPixelY) - static_cast<CoordT>(1);

            return {static_cast<CoordT>(aPixelX), y};
        }
        else if constexpr (mirrorAxis == MirrorAxis::Vertical)
        {
            CoordT x = static_cast<CoordT>(roiSize.x) - static_cast<CoordT>(aPixelX) - static_cast<CoordT>(1);

            return {x, static_cast<CoordT>(aPixelX)};
        }
        else if constexpr (mirrorAxis == MirrorAxis::Both)
        {
            CoordT x = static_cast<CoordT>(roiSize.x) - static_cast<CoordT>(aPixelX) - static_cast<CoordT>(1);
            CoordT y = static_cast<CoordT>(roiSize.y) - static_cast<CoordT>(aPixelY) - static_cast<CoordT>(1);

            return {x, y};
        }
        else
        {
            static_assert(mirrorAxis != MirrorAxis::Horizontal, "Unknown MirrorAxis.");
        }
    }
};

template <typename CoordT> struct TransformerMap
{
    const Vector2<CoordT> *RESTRICT PointerMap;
    const size_t MapPitch;

#pragma region Constructors
    TransformerMap(const Vector2<CoordT> *aPointerMap, size_t aMapPitch) : PointerMap(aPointerMap), MapPitch(aMapPitch)
    {
    }
#pragma endregion

    DEVICE_CODE Vector2<CoordT> operator()(int aPixelX, int aPixelY) const
    {
        return *gotoPtr(PointerMap, MapPitch, aPixelX, aPixelY);
    }
};

template <typename CoordT> struct TransformerMap2
{
    const CoordT *RESTRICT PointerMapX;
    const CoordT *RESTRICT PointerMapY;
    const size_t MapPitchX;
    const size_t MapPitchY;

#pragma region Constructors
    TransformerMap2(const CoordT *aPointerMapX, size_t aMapPitchX, const CoordT *aPointerMapY, size_t aMapPitchY)
        : PointerMapX(aPointerMapX), PointerMapY(aPointerMapY), MapPitchX(aMapPitchX), MapPitchY(aMapPitchY)
    {
    }
#pragma endregion

    DEVICE_CODE Vector2<CoordT> operator()(int aPixelX, int aPixelY) const
    {
        return {*gotoPtr(PointerMapX, MapPitchX, aPixelX, aPixelY), *gotoPtr(PointerMapY, MapPitchY, aPixelX, aPixelY)};
    }
};

template <typename CoordT> struct TransformerAffine
{
    const AffineTransformation<CoordT> AffineTransform;

#pragma region Constructors
    TransformerAffine(const AffineTransformation<CoordT> &aAffineTransform) : AffineTransform(aAffineTransform)
    {
    }
#pragma endregion

    DEVICE_CODE Vector2<CoordT> operator()(int aPixelX, int aPixelY) const
    {
        return AffineTransform * Vector2<CoordT>(aPixelX, aPixelY);
    }
};

template <typename CoordT> struct TransformerPerspective
{
    const Matrix<CoordT> PerspectiveTransform;

#pragma region Constructors
    TransformerPerspective(const Matrix<CoordT> &aPerspectiveTransform) : PerspectiveTransform(aPerspectiveTransform)
    {
    }
#pragma endregion

    DEVICE_CODE Vector2<CoordT> operator()(int aPixelX, int aPixelY) const
    {
        return PerspectiveTransform * Vector2<CoordT>(aPixelX, aPixelY);
    }
};

template <typename CoordT> struct TransformerResize
{
    const Vector2<CoordT> Scale;
    const Vector2<CoordT> Shift;

#pragma region Constructors
    TransformerResize(const Vector2<CoordT> &aScale, const Vector2<CoordT> &aShift) : Scale(aScale), Shift(aShift)
    {
    }
#pragma endregion

    DEVICE_CODE Vector2<CoordT> operator()(int aPixelX, int aPixelY) const
    {
        return Vector2<CoordT>(aPixelX, aPixelY) * Scale - Shift;
    }
};

template <typename CoordT> struct TransformerShift
{
    const Vector2<CoordT> Shift;

#pragma region Constructors
    TransformerShift(const Vector2<CoordT> &aShift) : Shift(aShift)
    {
    }
#pragma endregion

    DEVICE_CODE Vector2<CoordT> operator()(int aPixelX, int aPixelY) const
    {
        return Vector2<CoordT>(aPixelX, aPixelY) - Shift;
    }
};
} // namespace opp::image

#undef STD