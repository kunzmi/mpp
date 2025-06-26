#pragma once
#include "imageFunctors.h"
#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/unary_operators.h>
#include <common/defines.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/numberTypes.h>
#include <common/mpp_defs.h>
#include <common/roundFunctor.h>
#include <common/tupel.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>

// disable warning for pragma unroll when compiling with host compiler:
#include <common/disableWarningsBegin.h>

namespace mpp::image
{
/// <summary>
/// Computes an output pixel from one src array -&gt; dst pixel
/// </summary>
/// <typeparam name="SrcT"></typeparam>
/// <typeparam name="ComputeT"></typeparam>
/// <typeparam name="DstT"></typeparam>
/// <typeparam name="operation"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename SrcT, typename ComputeT, typename DstT, typename operation,
          RoundingMode roundingMode = RoundingMode::NearestTiesAwayFromZero, typename ComputeT_SIMD = voidType,
          typename operation_SIMD = voidType>
struct SrcFunctor : public ImageFunctor<false>
{
    const SrcT *RESTRICT Src1;
    size_t SrcPitch1;

    [[no_unique_address]] operation Op;
    [[no_unique_address]] operation_SIMD OpSIMD;

    [[no_unique_address]] RoundFunctor<roundingMode, ComputeT> round;
    [[no_unique_address]] RoundFunctor<
        roundingMode, same_vector_size_different_type_t<ComputeT, complex_basetype_t<remove_vector_t<ComputeT>>>>
        roundCplx2Real;

#pragma region Constructors
    SrcFunctor()
    {
    }

    SrcFunctor(const SrcT *aSrc1, size_t aSrcPitch1, operation aOp) : Src1(aSrc1), SrcPitch1(aSrcPitch1), Op(aOp)
    {
    }

    SrcFunctor(const SrcT *aSrc1, size_t aSrcPitch1, operation aOp, operation_SIMD aOpSIMD)
        : Src1(aSrc1), SrcPitch1(aSrcPitch1), Op(aOp), OpSIMD(aOpSIMD)
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    /// <summary>
    /// Returns true if the value has been successfully set
    /// </summary>
    DEVICE_CODE bool operator()(int aPixelX, int aPixelY, DstT &aDst) const
        requires std::same_as<ComputeT, DstT>
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        Op(static_cast<ComputeT>(*pixelSrc1), aDst);
        return true;
    }

    // for copy sub-channel and dup (SrcT == ComputeT)
    DEVICE_CODE bool operator()(int aPixelX, int aPixelY, DstT &aDst) const
        requires std::same_as<remove_vector_t<ComputeT>, remove_vector_t<DstT>> &&
                 (vector_size_v<ComputeT> != vector_size_v<DstT>)
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        Op(*pixelSrc1, aDst);
        return true;
    }

    /// <summary>
    /// Returns true if the value has been successfully set
    /// </summary>
    DEVICE_CODE bool operator()(int aPixelX, int aPixelY, DstT &aDst) const
        requires(!std::same_as<ComputeT, DstT>) && (!(ComplexVector<ComputeT> && RealVector<DstT>)) &&
                (!(RealVector<ComputeT> && ComplexVector<DstT>)) && (vector_size_v<ComputeT> == vector_size_v<DstT>)
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        ComputeT temp;
        Op(static_cast<ComputeT>(*pixelSrc1), temp);
        round(temp); // NOP for integer ComputeT
        // DstT constructor will clamp temp to value range of DstT
        aDst = static_cast<DstT>(temp);
        return true;
    }

    /// <summary>
    /// Returns true if the value has been successfully set
    /// </summary>
    DEVICE_CODE bool operator()(int aPixelX, int aPixelY, DstT &aDst) const
        requires(!std::same_as<ComputeT, DstT>) && ((ComplexVector<ComputeT> && RealVector<DstT>))
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        same_vector_size_different_type_t<ComputeT, complex_basetype_t<remove_vector_t<ComputeT>>> temp;
        Op(static_cast<ComputeT>(*pixelSrc1), temp);
        if constexpr (!RealOrComplexFloatingVector<DstT>)
        {
            roundCplx2Real(temp); // NOP for integer ComputeT
        }
        // DstT constructor will clamp temp to value range of DstT
        aDst = static_cast<DstT>(temp);
        return true;
    }

    // Needed for the conversion real->Complex
    DEVICE_CODE bool operator()(int aPixelX, int aPixelY, DstT &aDst) const
        requires(!std::same_as<ComputeT, DstT>) && ((RealVector<ComputeT> && ComplexVector<DstT>))
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        Op(*pixelSrc1, aDst);
        return true;
    }
#pragma endregion

#pragma region run SIMD on pixel tupel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst) const
        requires std::same_as<ComputeT_SIMD, DstT>
    {
        static_assert(OpSIMD.has_simd, "Trying to run a SIMD operation that is not implemented for this type.");
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::Load(pixelSrc1);

        OpSIMD(tupelSrc1, aDst);
    }
#pragma endregion

#pragma region run sequential on pixel tupel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst) const
        requires std::same_as<ComputeT, DstT> && //
                 std::same_as<ComputeT_SIMD, voidType>
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::Load(pixelSrc1);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            Op(static_cast<ComputeT>(tupelSrc1.value[i]), aDst.value[i]);
        }
    }

    // for copy sub-channel and dup (SrcT == ComputeT)
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst) const
        requires std::same_as<remove_vector_t<ComputeT>, remove_vector_t<DstT>> && //
                 (vector_size_v<ComputeT> != vector_size_v<DstT>) &&               //
                 std::same_as<ComputeT_SIMD, voidType>
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::Load(pixelSrc1);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            Op(tupelSrc1.value[i], aDst.value[i]);
        }
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst) const
        requires(!std::same_as<ComputeT, DstT>) && //
                std::same_as<ComputeT_SIMD, voidType> && (!(ComplexVector<ComputeT> && RealVector<DstT>)) &&
                (!(RealVector<ComputeT> && ComplexVector<DstT>)) && (vector_size_v<ComputeT> == vector_size_v<DstT>)
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::Load(pixelSrc1);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            ComputeT temp;
            Op(static_cast<ComputeT>(tupelSrc1.value[i]), temp);
            round(temp); // NOP for integer ComputeT
            // DstT constructor will clamp temp to value range of DstT
            aDst.value[i] = static_cast<DstT>(temp);
        }
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst) const
        requires(!std::same_as<ComputeT, DstT>) && //
                std::same_as<ComputeT_SIMD, voidType> && ((ComplexVector<ComputeT> && RealVector<DstT>))
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::Load(pixelSrc1);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            same_vector_size_different_type_t<ComputeT, complex_basetype_t<remove_vector_t<ComputeT>>> temp;
            Op(static_cast<ComputeT>(tupelSrc1.value[i]), temp);
            if constexpr (!RealOrComplexFloatingVector<DstT>)
            {
                roundCplx2Real(temp); // NOP for integer ComputeT
            }
            // DstT constructor will clamp temp to value range of DstT
            aDst.value[i] = static_cast<DstT>(temp);
        }
    }

    // Needed for the conversion real->Complex
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst) const
        requires(!std::same_as<ComputeT, DstT>) && //
                std::same_as<ComputeT_SIMD, voidType> && ((RealVector<ComputeT> && ComplexVector<DstT>))
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::Load(pixelSrc1);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            Op(tupelSrc1.value[i], aDst.value[i]);
        }
    }
#pragma endregion
};
} // namespace mpp::image
#include <common/disableWarningsEnd.h>
