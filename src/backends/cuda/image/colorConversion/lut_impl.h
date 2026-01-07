#include "lut.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/colorConversion/color_operators.h>
#include <common/defines.h>
#include <common/image/functors/inplaceFunctor.h>
#include <common/image/functors/srcFunctor.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/mpp_defs.h>
#include <common/safeCast.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace mpp::cuda;

namespace mpp::image::cuda
{

template <typename TT> struct GetLUTComputeT
{
    using type = double;
};
template <> struct GetLUTComputeT<HalfFp16>
{
    using type = float;
};
template <> struct GetLUTComputeT<BFloat16>
{
    using type = float;
};

template <typename SrcDstT>
void InvokeLutPaletteSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst,
                         const SrcDstT *aPalette, int aBitSize, const Size2D &aSize,
                         const mpp::cuda::StreamCtx &aStreamCtx)
{
    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));
    using SrcT           = SrcDstT;
    using DstT           = SrcDstT;
    using ComputeT       = SrcDstT;
    using LutT           = SrcDstT;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using lutSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT>,
                              RoundingMode::None>;

    const mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT> op(aPalette, indexBound);

    const lutSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLutPaletteSrc_For(typeSrcIsTypeDst)                                                           \
    template void InvokeLutPaletteSrc<typeSrcIsTypeDst>(                                                               \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        const typeSrcIsTypeDst *aPalette, int aBitSize, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLutPaletteSrc(type) InstantiateInvokeLutPaletteSrc_For(Pixel##type##C1);

#pragma endregion

template <typename SrcDstT>
void InvokeLutPaletteInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, const SrcDstT *aPalette, int aBitSize,
                             const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));
    using SrcT           = SrcDstT;
    using DstT           = SrcDstT;
    using ComputeT       = SrcDstT;
    using LutT           = SrcDstT;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using lutSrc = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT>,
                                  RoundingMode::None>;

    const mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT> op(aPalette, indexBound);

    const lutSrc functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLutPaletteInplace_For(typeSrcIsTypeDst)                                                       \
    template void InvokeLutPaletteInplace<typeSrcIsTypeDst>(                                                           \
        typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1, const typeSrcIsTypeDst *aPalette, int aBitSize,             \
        const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLutPaletteInplace(type) InstantiateInvokeLutPaletteInplace_For(Pixel##type##C1);

#pragma endregion

template <typename SrcDstT>
void InvokeLutPaletteSrc33(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector3<remove_vector_t<SrcDstT>> *aDst,
                           size_t aPitchDst, const Vector3<remove_vector_t<SrcDstT>> *aPalette, int aBitSize,
                           const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));
    using SrcT           = SrcDstT;
    using DstT           = Vector3<remove_vector_t<SrcDstT>>;
    using ComputeT       = SrcDstT;
    using LutT           = Vector3<remove_vector_t<SrcDstT>>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using lutSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT>,
                              RoundingMode::None>;

    const mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT> op(aPalette, indexBound);

    const lutSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLutPaletteSrc33_For(typeSrcIsTypeDst)                                                         \
    template void InvokeLutPaletteSrc33<typeSrcIsTypeDst>(                                                             \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector3<remove_vector_t<typeSrcIsTypeDst>> *aDst,            \
        size_t aPitchDst, const Vector3<remove_vector_t<typeSrcIsTypeDst>> *aPalette, int aBitSize,                    \
        const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLutPaletteSrc33(type) InstantiateInvokeLutPaletteSrc33_For(Pixel##type##C1);

#pragma endregion

template <typename SrcDstT>
void InvokeLutPaletteSrc34A(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector3<remove_vector_t<SrcDstT>> *aDst,
                            size_t aPitchDst, const Vector4A<remove_vector_t<SrcDstT>> *aPalette, int aBitSize,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));
    using SrcT           = SrcDstT;
    using DstT           = Vector3<remove_vector_t<SrcDstT>>;
    using ComputeT       = SrcDstT;
    using LutT           = Vector4A<remove_vector_t<SrcDstT>>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using lutSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT>,
                              RoundingMode::None>;

    const mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT> op(aPalette, indexBound);

    const lutSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLutPaletteSrc34A_For(typeSrcIsTypeDst)                                                        \
    template void InvokeLutPaletteSrc34A<typeSrcIsTypeDst>(                                                            \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector3<remove_vector_t<typeSrcIsTypeDst>> *aDst,            \
        size_t aPitchDst, const Vector4A<remove_vector_t<typeSrcIsTypeDst>> *aPalette, int aBitSize,                   \
        const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLutPaletteSrc34A(type) InstantiateInvokeLutPaletteSrc34A_For(Pixel##type##C1);

#pragma endregion

template <typename SrcDstT>
void InvokeLutPaletteSrc4A3(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector4A<remove_vector_t<SrcDstT>> *aDst,
                            size_t aPitchDst, const Vector3<remove_vector_t<SrcDstT>> *aPalette, int aBitSize,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));
    using SrcT           = SrcDstT;
    using DstT           = Vector4A<remove_vector_t<SrcDstT>>;
    using ComputeT       = SrcDstT;
    using LutT           = Vector3<remove_vector_t<SrcDstT>>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using lutSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT>,
                              RoundingMode::None>;

    const mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT> op(aPalette, indexBound);

    const lutSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLutPaletteSrc4A3_For(typeSrcIsTypeDst)                                                        \
    template void InvokeLutPaletteSrc4A3<typeSrcIsTypeDst>(                                                            \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector4A<remove_vector_t<typeSrcIsTypeDst>> *aDst,           \
        size_t aPitchDst, const Vector3<remove_vector_t<typeSrcIsTypeDst>> *aPalette, int aBitSize,                    \
        const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLutPaletteSrc4A3(type) InstantiateInvokeLutPaletteSrc4A3_For(Pixel##type##C1);

#pragma endregion

template <typename SrcDstT>
void InvokeLutPaletteSrc4A4A(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector4A<remove_vector_t<SrcDstT>> *aDst,
                             size_t aPitchDst, const Vector4A<remove_vector_t<SrcDstT>> *aPalette, int aBitSize,
                             const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));
    using SrcT           = SrcDstT;
    using DstT           = Vector4A<remove_vector_t<SrcDstT>>;
    using ComputeT       = SrcDstT;
    using LutT           = Vector4A<remove_vector_t<SrcDstT>>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using lutSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT>,
                              RoundingMode::None>;

    const mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT> op(aPalette, indexBound);

    const lutSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLutPaletteSrc4A4A_For(typeSrcIsTypeDst)                                                       \
    template void InvokeLutPaletteSrc4A4A<typeSrcIsTypeDst>(                                                           \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector4A<remove_vector_t<typeSrcIsTypeDst>> *aDst,           \
        size_t aPitchDst, const Vector4A<remove_vector_t<typeSrcIsTypeDst>> *aPalette, int aBitSize,                   \
        const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLutPaletteSrc4A4A(type) InstantiateInvokeLutPaletteSrc4A4A_For(Pixel##type##C1);

#pragma endregion

template <typename SrcDstT>
void InvokeLutPaletteSrc44(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector4<remove_vector_t<SrcDstT>> *aDst,
                           size_t aPitchDst, const Vector4<remove_vector_t<SrcDstT>> *aPalette, int aBitSize,
                           const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));
    using SrcT           = SrcDstT;
    using DstT           = Vector4<remove_vector_t<SrcDstT>>;
    using ComputeT       = SrcDstT;
    using LutT           = Vector4<remove_vector_t<SrcDstT>>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using lutSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT>,
                              RoundingMode::None>;

    const mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT> op(aPalette, indexBound);

    const lutSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLutPaletteSrc44_For(typeSrcIsTypeDst)                                                         \
    template void InvokeLutPaletteSrc44<typeSrcIsTypeDst>(                                                             \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector4<remove_vector_t<typeSrcIsTypeDst>> *aDst,            \
        size_t aPitchDst, const Vector4<remove_vector_t<typeSrcIsTypeDst>> *aPalette, int aBitSize,                    \
        const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLutPaletteSrc44(type) InstantiateInvokeLutPaletteSrc44_For(Pixel##type##C1);

#pragma endregion

template <typename SrcDstT>
void InvokeLutPaletteSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst,
                         const Vector1<remove_vector_t<SrcDstT>> *const *aPalette, int aBitSize, const Size2D &aSize,
                         const mpp::cuda::StreamCtx &aStreamCtx)
{
    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcDstT)>::value;

    if constexpr (vector_active_size_v<SrcDstT> == 2)
    {
        using lutSrc = SrcFunctor<TupelSize, SrcDstT, SrcDstT, SrcDstT,
                                  mpp::image::LUTPalettePlanar2WithBounds<SrcDstT>, RoundingMode::None>;

        const mpp::image::LUTPalettePlanar2WithBounds<SrcDstT> op(
            reinterpret_cast<const remove_vector_t<SrcDstT> *>(aPalette[0]),
            reinterpret_cast<const remove_vector_t<SrcDstT> *>(aPalette[1]), indexBound);
        const lutSrc functor(aSrc1, aPitchSrc1, op);
        InvokeForEachPixelKernelDefault<SrcDstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
    else if constexpr (vector_active_size_v<SrcDstT> == 3)
    {
        using lutSrc = SrcFunctor<TupelSize, SrcDstT, SrcDstT, SrcDstT,
                                  mpp::image::LUTPalettePlanar3WithBounds<SrcDstT>, RoundingMode::None>;

        const mpp::image::LUTPalettePlanar3WithBounds<SrcDstT> op(
            reinterpret_cast<const remove_vector_t<SrcDstT> *>(aPalette[0]),
            reinterpret_cast<const remove_vector_t<SrcDstT> *>(aPalette[1]),
            reinterpret_cast<const remove_vector_t<SrcDstT> *>(aPalette[2]), indexBound);
        const lutSrc functor(aSrc1, aPitchSrc1, op);
        InvokeForEachPixelKernelDefault<SrcDstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
    else
    {
        using lutSrc = SrcFunctor<TupelSize, SrcDstT, SrcDstT, SrcDstT,
                                  mpp::image::LUTPalettePlanar4WithBounds<SrcDstT>, RoundingMode::None>;

        const mpp::image::LUTPalettePlanar4WithBounds<SrcDstT> op(
            reinterpret_cast<const remove_vector_t<SrcDstT> *>(aPalette[0]),
            reinterpret_cast<const remove_vector_t<SrcDstT> *>(aPalette[1]),
            reinterpret_cast<const remove_vector_t<SrcDstT> *>(aPalette[2]),
            reinterpret_cast<const remove_vector_t<SrcDstT> *>(aPalette[3]), indexBound);
        const lutSrc functor(aSrc1, aPitchSrc1, op);
        InvokeForEachPixelKernelDefault<SrcDstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
#define InstantiateInvokeLutPaletteSrcP_For(typeSrcIsTypeDst)                                                          \
    template void InvokeLutPaletteSrc<typeSrcIsTypeDst>(                                                               \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *const *aPalette, int aBitSize, const Size2D &aSize,          \
        const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLutPaletteSrcP(type)                                                              \
    InstantiateInvokeLutPaletteSrcP_For(Pixel##type##C2);                                                              \
    InstantiateInvokeLutPaletteSrcP_For(Pixel##type##C3);                                                              \
    InstantiateInvokeLutPaletteSrcP_For(Pixel##type##C4);                                                              \
    InstantiateInvokeLutPaletteSrcP_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeLutPaletteInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1,
                             const Vector1<remove_vector_t<SrcDstT>> *const *aPalette, int aBitSize,
                             const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    const int indexBound       = static_cast<int>(1u << static_cast<uint>(aBitSize));
    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcDstT)>::value;

    if constexpr (vector_active_size_v<SrcDstT> == 2)
    {
        using lutSrc = InplaceFunctor<TupelSize, SrcDstT, SrcDstT, mpp::image::LUTPalettePlanar2WithBounds<SrcDstT>,
                                      RoundingMode::None>;

        const mpp::image::LUTPalettePlanar2WithBounds<SrcDstT> op(
            reinterpret_cast<const remove_vector_t<SrcDstT> *>(aPalette[0]),
            reinterpret_cast<const remove_vector_t<SrcDstT> *>(aPalette[1]), indexBound);
        const lutSrc functor(op);
        InvokeForEachPixelKernelDefault<SrcDstT, TupelSize, lutSrc>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx,
                                                                    functor);
    }
    else if constexpr (vector_active_size_v<SrcDstT> == 3)
    {
        using lutSrc = InplaceFunctor<TupelSize, SrcDstT, SrcDstT, mpp::image::LUTPalettePlanar3WithBounds<SrcDstT>,
                                      RoundingMode::None>;

        const mpp::image::LUTPalettePlanar3WithBounds<SrcDstT> op(
            reinterpret_cast<const remove_vector_t<SrcDstT> *>(aPalette[0]),
            reinterpret_cast<const remove_vector_t<SrcDstT> *>(aPalette[1]),
            reinterpret_cast<const remove_vector_t<SrcDstT> *>(aPalette[2]), indexBound);
        const lutSrc functor(op);
        InvokeForEachPixelKernelDefault<SrcDstT, TupelSize, lutSrc>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx,
                                                                    functor);
    }
    else
    {
        using lutSrc = InplaceFunctor<TupelSize, SrcDstT, SrcDstT, mpp::image::LUTPalettePlanar4WithBounds<SrcDstT>,
                                      RoundingMode::None>;

        const mpp::image::LUTPalettePlanar4WithBounds<SrcDstT> op(
            reinterpret_cast<const remove_vector_t<SrcDstT> *>(aPalette[0]),
            reinterpret_cast<const remove_vector_t<SrcDstT> *>(aPalette[1]),
            reinterpret_cast<const remove_vector_t<SrcDstT> *>(aPalette[2]),
            reinterpret_cast<const remove_vector_t<SrcDstT> *>(aPalette[3]), indexBound);
        const lutSrc functor(op);
        InvokeForEachPixelKernelDefault<SrcDstT, TupelSize, lutSrc>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx,
                                                                    functor);
    }
}

#pragma region Instantiate
#define InstantiateInvokeLutPaletteInplaceP_For(typeSrcIsTypeDst)                                                      \
    template void InvokeLutPaletteInplace<typeSrcIsTypeDst>(                                                           \
        typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1,                                                             \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *const *aPalette, int aBitSize, const Size2D &aSize,          \
        const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLutPaletteInplaceP(type)                                                          \
    InstantiateInvokeLutPaletteInplaceP_For(Pixel##type##C2);                                                          \
    InstantiateInvokeLutPaletteInplaceP_For(Pixel##type##C3);                                                          \
    InstantiateInvokeLutPaletteInplaceP_For(Pixel##type##C4);                                                          \
    InstantiateInvokeLutPaletteInplaceP_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeLutPaletteSrc16u(const SrcDstT *aSrc1, size_t aPitchSrc1, Pixel8uC1 *aDst, size_t aPitchDst,
                            const Pixel8uC1 *aPalette, int aBitSize, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx)
{
    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));
    using SrcT           = SrcDstT;
    using DstT           = Pixel8uC1;
    using ComputeT       = SrcDstT;
    using LutT           = Pixel8uC1;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using lutSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT>,
                              RoundingMode::None, voidType, voidType, true>;

    const mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT> op(aPalette, indexBound);

    const lutSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLutPaletteSrc8uC1_For(typeSrcIsTypeDst)                                                       \
    template void InvokeLutPaletteSrc16u<typeSrcIsTypeDst>(                                                            \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Pixel8uC1 *aDst, size_t aPitchDst,                           \
        const Pixel8uC1 *aPalette, int aBitSize, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLutPaletteSrc8uC1(type) InstantiateInvokeLutPaletteSrc8uC1_For(Pixel##type##C1);

#pragma endregion

template <typename SrcDstT>
void InvokeLutPaletteSrc16u(const SrcDstT *aSrc1, size_t aPitchSrc1, Pixel8uC3 *aDst, size_t aPitchDst,
                            const Pixel8uC3 *aPalette, int aBitSize, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx)
{
    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));
    using SrcT           = SrcDstT;
    using DstT           = Pixel8uC3;
    using ComputeT       = SrcDstT;
    using LutT           = Pixel8uC3;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using lutSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT>,
                              RoundingMode::None, voidType, voidType, true>;

    const mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT> op(aPalette, indexBound);

    const lutSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLutPaletteSrc8uC3_For(typeSrcIsTypeDst)                                                       \
    template void InvokeLutPaletteSrc16u<typeSrcIsTypeDst>(                                                            \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Pixel8uC3 *aDst, size_t aPitchDst,                           \
        const Pixel8uC3 *aPalette, int aBitSize, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLutPaletteSrc8uC3(type) InstantiateInvokeLutPaletteSrc8uC3_For(Pixel##type##C1);

#pragma endregion

template <typename SrcDstT>
void InvokeLutPaletteSrc16u(const SrcDstT *aSrc1, size_t aPitchSrc1, Pixel8uC4 *aDst, size_t aPitchDst,
                            const Pixel8uC4 *aPalette, int aBitSize, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx)
{
    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));
    using SrcT           = SrcDstT;
    using DstT           = Pixel8uC4;
    using ComputeT       = SrcDstT;
    using LutT           = Pixel8uC4;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using lutSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT>,
                              RoundingMode::None, voidType, voidType, true>;

    const mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT> op(aPalette, indexBound);

    const lutSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLutPaletteSrc8uC4_For(typeSrcIsTypeDst)                                                       \
    template void InvokeLutPaletteSrc16u<typeSrcIsTypeDst>(                                                            \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Pixel8uC4 *aDst, size_t aPitchDst,                           \
        const Pixel8uC4 *aPalette, int aBitSize, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLutPaletteSrc8uC4(type) InstantiateInvokeLutPaletteSrc8uC4_For(Pixel##type##C1);

#pragma endregion

template <typename SrcDstT>
void InvokeLutPaletteSrc16u(const SrcDstT *aSrc1, size_t aPitchSrc1, Pixel8uC4A *aDst, size_t aPitchDst,
                            const Pixel8uC4A *aPalette, int aBitSize, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx)
{
    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));
    using SrcT           = SrcDstT;
    using DstT           = Pixel8uC4A;
    using ComputeT       = SrcDstT;
    using LutT           = Pixel8uC4A;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using lutSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT>,
                              RoundingMode::None, voidType, voidType, true>;

    const mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT> op(aPalette, indexBound);

    const lutSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLutPaletteSrc8uC4A_For(typeSrcIsTypeDst)                                                      \
    template void InvokeLutPaletteSrc16u<typeSrcIsTypeDst>(                                                            \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Pixel8uC4A *aDst, size_t aPitchDst,                          \
        const Pixel8uC4A *aPalette, int aBitSize, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLutPaletteSrc8uC4A(type) InstantiateInvokeLutPaletteSrc8uC4A_For(Pixel##type##C1);

#pragma endregion

template <typename SrcDstT>
void InvokeLutSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, const Pixel32fC1 *aLevels,
                  const Pixel32fC1 *aValues, const int *aAccelerator, int aLutSize, int aAcceleratorSize,
                  InterpolationMode aInterpolationMode, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT        = SrcDstT;
    using DstT        = SrcDstT;
    using ComputeT    = SrcDstT;
    using LutComputeT = typename GetLUTComputeT<remove_vector_t<SrcDstT>>::type;
    using LutT        = float;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    if (aInterpolationMode == InterpolationMode::NearestNeighbor)
    {
        using opT            = mpp::image::LUT1Channel<SrcT, LutT, LutComputeT, InterpolationMode::NearestNeighbor>;
        using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
        using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

        const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels), reinterpret_cast<const float *>(aValues),
                                     aAccelerator, aLutSize, aAcceleratorSize);
        const opT op(lutSrc0);
        const lutSrc functor(aSrc1, aPitchSrc1, op);
        InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
    else if (aInterpolationMode == InterpolationMode::Linear)
    {
        using opT            = mpp::image::LUT1Channel<SrcT, LutT, LutComputeT, InterpolationMode::Linear>;
        using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
        using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

        const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels), reinterpret_cast<const float *>(aValues),
                                     aAccelerator, aLutSize, aAcceleratorSize);
        const opT op(lutSrc0);
        const lutSrc functor(aSrc1, aPitchSrc1, op);
        InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
    else if (aInterpolationMode == InterpolationMode::CubicLagrange)
    {
        using opT            = mpp::image::LUT1Channel<SrcT, LutT, LutComputeT, InterpolationMode::CubicLagrange>;
        using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
        using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

        const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels), reinterpret_cast<const float *>(aValues),
                                     aAccelerator, aLutSize, aAcceleratorSize);
        const opT op(lutSrc0);
        const lutSrc functor(aSrc1, aPitchSrc1, op);
        InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
    else
    {
        throw INVALIDARGUMENT(aInterpolationMode, "Unsupported interpolation mode. Only NearestNeighbor, Linear and "
                                                  "CubicLagrange are supported, but provided aInterpolationMode is "
                                                      << aInterpolationMode);
    }
}

#pragma region Instantiate
#define InstantiateInvokeLutSrc1_For(typeSrcIsTypeDst)                                                                 \
    template void InvokeLutSrc<typeSrcIsTypeDst>(const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                     \
                                                 typeSrcIsTypeDst *aDst, size_t aPitchDst, const Pixel32fC1 *aLevels,  \
                                                 const Pixel32fC1 *aValues, const int *aAccelerator, int aLutSize,     \
                                                 int aAcceleratorSize, InterpolationMode aInterpolationMode,           \
                                                 const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLutSrc1(type) InstantiateInvokeLutSrc1_For(Pixel##type##C1);

#pragma endregion

template <typename SrcDstT>
void InvokeLutInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, const Pixel32fC1 *aLevels, const Pixel32fC1 *aValues,
                      const int *aAccelerator, int aLutSize, int aAcceleratorSize, InterpolationMode aInterpolationMode,
                      const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT        = SrcDstT;
    using DstT        = SrcDstT;
    using ComputeT    = SrcDstT;
    using LutComputeT = typename GetLUTComputeT<remove_vector_t<SrcDstT>>::type;
    using LutT        = float;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    if (aInterpolationMode == InterpolationMode::NearestNeighbor)
    {
        using opT            = mpp::image::LUT1Channel<SrcT, LutT, LutComputeT, InterpolationMode::NearestNeighbor>;
        using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
        using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

        const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels), reinterpret_cast<const float *>(aValues),
                                     aAccelerator, aLutSize, aAcceleratorSize);
        const opT op(lutSrc0);
        const lutSrc functor(op);
        InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
    }
    else if (aInterpolationMode == InterpolationMode::Linear)
    {
        using opT            = mpp::image::LUT1Channel<SrcT, LutT, LutComputeT, InterpolationMode::Linear>;
        using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
        using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

        const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels), reinterpret_cast<const float *>(aValues),
                                     aAccelerator, aLutSize, aAcceleratorSize);
        const opT op(lutSrc0);
        const lutSrc functor(op);
        InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
    }
    else if (aInterpolationMode == InterpolationMode::CubicLagrange)
    {
        using opT            = mpp::image::LUT1Channel<SrcT, LutT, LutComputeT, InterpolationMode::CubicLagrange>;
        using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
        using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

        const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels), reinterpret_cast<const float *>(aValues),
                                     aAccelerator, aLutSize, aAcceleratorSize);
        const opT op(lutSrc0);
        const lutSrc functor(op);
        InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
    }
    else
    {
        throw INVALIDARGUMENT(aInterpolationMode, "Unsupported interpolation mode. Only NearestNeighbor, Linear and "
                                                  "CubicLagrange are supported, but provided aInterpolationMode is "
                                                      << aInterpolationMode);
    }
}

#pragma region Instantiate
#define InstantiateInvokeLutInplace1_For(typeSrcIsTypeDst)                                                             \
    template void InvokeLutInplace<typeSrcIsTypeDst>(                                                                  \
        typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1, const Pixel32fC1 *aLevels, const Pixel32fC1 *aValues,       \
        const int *aAccelerator, int aLutSize, int aAcceleratorSize, InterpolationMode aInterpolationMode,             \
        const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLutInplace1(type) InstantiateInvokeLutInplace1_For(Pixel##type##C1);

#pragma endregion

template <typename SrcDstT>
void InvokeLutSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst,
                  const Pixel32fC1 *const *aLevels, const Pixel32fC1 *const *aValues, const int *const *aAccelerator,
                  int const *aLutSize, int const *aAcceleratorSize, InterpolationMode aInterpolationMode,
                  const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT        = SrcDstT;
    using DstT        = SrcDstT;
    using ComputeT    = SrcDstT;
    using LutComputeT = typename GetLUTComputeT<remove_vector_t<SrcDstT>>::type;
    using LutT        = float;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    if constexpr (vector_active_size_v<SrcDstT> == 2)
    {
        if (aInterpolationMode == InterpolationMode::NearestNeighbor)
        {
            using opT            = mpp::image::LUT2Channel<SrcT, LutT, LutComputeT, InterpolationMode::NearestNeighbor>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);

            const opT op(lutSrc0, lutSrc1);
            const lutSrc functor(aSrc1, aPitchSrc1, op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        else if (aInterpolationMode == InterpolationMode::Linear)
        {
            using opT            = mpp::image::LUT2Channel<SrcT, LutT, LutComputeT, InterpolationMode::Linear>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);

            const opT op(lutSrc0, lutSrc1);
            const lutSrc functor(aSrc1, aPitchSrc1, op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        else
        {
            using opT            = mpp::image::LUT2Channel<SrcT, LutT, LutComputeT, InterpolationMode::CubicLagrange>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);

            const opT op(lutSrc0, lutSrc1);
            const lutSrc functor(aSrc1, aPitchSrc1, op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
    }
    else if constexpr (vector_active_size_v<SrcDstT> == 3)
    {
        if (aInterpolationMode == InterpolationMode::NearestNeighbor)
        {
            using opT            = mpp::image::LUT3Channel<SrcT, LutT, LutComputeT, InterpolationMode::NearestNeighbor>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);

            const opT op(lutSrc0, lutSrc1, lutSrc2);
            const lutSrc functor(aSrc1, aPitchSrc1, op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        else if (aInterpolationMode == InterpolationMode::Linear)
        {
            using opT            = mpp::image::LUT3Channel<SrcT, LutT, LutComputeT, InterpolationMode::Linear>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);

            const opT op(lutSrc0, lutSrc1, lutSrc2);
            const lutSrc functor(aSrc1, aPitchSrc1, op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        else
        {
            using opT            = mpp::image::LUT3Channel<SrcT, LutT, LutComputeT, InterpolationMode::CubicLagrange>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);

            const opT op(lutSrc0, lutSrc1, lutSrc2);
            const lutSrc functor(aSrc1, aPitchSrc1, op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
    }
    else
    {
        if (aInterpolationMode == InterpolationMode::NearestNeighbor)
        {
            using opT            = mpp::image::LUT4Channel<SrcT, LutT, LutComputeT, InterpolationMode::NearestNeighbor>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);
            const LUTChannelSrcT lutSrc3(reinterpret_cast<const float *>(aLevels[3]),
                                         reinterpret_cast<const float *>(aValues[3]), aAccelerator[3], aLutSize[3],
                                         aAcceleratorSize[3]);

            const opT op(lutSrc0, lutSrc1, lutSrc2, lutSrc3);
            const lutSrc functor(aSrc1, aPitchSrc1, op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        else if (aInterpolationMode == InterpolationMode::Linear)
        {
            using opT            = mpp::image::LUT4Channel<SrcT, LutT, LutComputeT, InterpolationMode::Linear>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);
            const LUTChannelSrcT lutSrc3(reinterpret_cast<const float *>(aLevels[3]),
                                         reinterpret_cast<const float *>(aValues[3]), aAccelerator[3], aLutSize[3],
                                         aAcceleratorSize[3]);

            const opT op(lutSrc0, lutSrc1, lutSrc2, lutSrc3);
            const lutSrc functor(aSrc1, aPitchSrc1, op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        else
        {
            using opT            = mpp::image::LUT4Channel<SrcT, LutT, LutComputeT, InterpolationMode::CubicLagrange>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);
            const LUTChannelSrcT lutSrc3(reinterpret_cast<const float *>(aLevels[3]),
                                         reinterpret_cast<const float *>(aValues[3]), aAccelerator[3], aLutSize[3],
                                         aAcceleratorSize[3]);

            const opT op(lutSrc0, lutSrc1, lutSrc2, lutSrc3);
            const lutSrc functor(aSrc1, aPitchSrc1, op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
#define InstantiateInvokeLutSrc_For(typeSrcIsTypeDst)                                                                  \
    template void InvokeLutSrc<typeSrcIsTypeDst>(                                                                      \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        const Pixel32fC1 *const *aLevels, const Pixel32fC1 *const *aValues, const int *const *aAccelerator,            \
        int const *aLutSize, int const *aAcceleratorSize, InterpolationMode aInterpolationMode, const Size2D &aSize,   \
        const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLutSrc(type)                                                                      \
    InstantiateInvokeLutSrc_For(Pixel##type##C2);                                                                      \
    InstantiateInvokeLutSrc_For(Pixel##type##C3);                                                                      \
    InstantiateInvokeLutSrc_For(Pixel##type##C4);                                                                      \
    InstantiateInvokeLutSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeLutInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, const Pixel32fC1 *const *aLevels,
                      const Pixel32fC1 *const *aValues, const int *const *aAccelerator, int const *aLutSize,
                      int const *aAcceleratorSize, InterpolationMode aInterpolationMode, const Size2D &aSize,
                      const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT        = SrcDstT;
    using DstT        = SrcDstT;
    using ComputeT    = SrcDstT;
    using LutComputeT = typename GetLUTComputeT<remove_vector_t<SrcDstT>>::type;
    using LutT        = float;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    if constexpr (vector_active_size_v<SrcDstT> == 2)
    {
        if (aInterpolationMode == InterpolationMode::NearestNeighbor)
        {
            using opT            = mpp::image::LUT2Channel<SrcT, LutT, LutComputeT, InterpolationMode::NearestNeighbor>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);

            const opT op(lutSrc0, lutSrc1);
            const lutSrc functor(op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx,
                                                                     functor);
        }
        else if (aInterpolationMode == InterpolationMode::Linear)
        {
            using opT            = mpp::image::LUT2Channel<SrcT, LutT, LutComputeT, InterpolationMode::Linear>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);

            const opT op(lutSrc0, lutSrc1);
            const lutSrc functor(op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx,
                                                                     functor);
        }
        else
        {
            using opT            = mpp::image::LUT2Channel<SrcT, LutT, LutComputeT, InterpolationMode::CubicLagrange>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);

            const opT op(lutSrc0, lutSrc1);
            const lutSrc functor(op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx,
                                                                     functor);
        }
    }
    else if constexpr (vector_active_size_v<SrcDstT> == 3)
    {
        if (aInterpolationMode == InterpolationMode::NearestNeighbor)
        {
            using opT            = mpp::image::LUT3Channel<SrcT, LutT, LutComputeT, InterpolationMode::NearestNeighbor>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);

            const opT op(lutSrc0, lutSrc1, lutSrc2);
            const lutSrc functor(op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx,
                                                                     functor);
        }
        else if (aInterpolationMode == InterpolationMode::Linear)
        {
            using opT            = mpp::image::LUT3Channel<SrcT, LutT, LutComputeT, InterpolationMode::Linear>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);

            const opT op(lutSrc0, lutSrc1, lutSrc2);
            const lutSrc functor(op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx,
                                                                     functor);
        }
        else
        {
            using opT            = mpp::image::LUT3Channel<SrcT, LutT, LutComputeT, InterpolationMode::CubicLagrange>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);

            const opT op(lutSrc0, lutSrc1, lutSrc2);
            const lutSrc functor(op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx,
                                                                     functor);
        }
    }
    else
    {
        if (aInterpolationMode == InterpolationMode::NearestNeighbor)
        {
            using opT            = mpp::image::LUT4Channel<SrcT, LutT, LutComputeT, InterpolationMode::NearestNeighbor>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);
            const LUTChannelSrcT lutSrc3(reinterpret_cast<const float *>(aLevels[3]),
                                         reinterpret_cast<const float *>(aValues[3]), aAccelerator[3], aLutSize[3],
                                         aAcceleratorSize[3]);

            const opT op(lutSrc0, lutSrc1, lutSrc2, lutSrc3);
            const lutSrc functor(op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx,
                                                                     functor);
        }
        else if (aInterpolationMode == InterpolationMode::Linear)
        {
            using opT            = mpp::image::LUT4Channel<SrcT, LutT, LutComputeT, InterpolationMode::Linear>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);
            const LUTChannelSrcT lutSrc3(reinterpret_cast<const float *>(aLevels[3]),
                                         reinterpret_cast<const float *>(aValues[3]), aAccelerator[3], aLutSize[3],
                                         aAcceleratorSize[3]);

            const opT op(lutSrc0, lutSrc1, lutSrc2, lutSrc3);
            const lutSrc functor(op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx,
                                                                     functor);
        }
        else
        {
            using opT            = mpp::image::LUT4Channel<SrcT, LutT, LutComputeT, InterpolationMode::CubicLagrange>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);
            const LUTChannelSrcT lutSrc3(reinterpret_cast<const float *>(aLevels[3]),
                                         reinterpret_cast<const float *>(aValues[3]), aAccelerator[3], aLutSize[3],
                                         aAcceleratorSize[3]);

            const opT op(lutSrc0, lutSrc1, lutSrc2, lutSrc3);
            const lutSrc functor(op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx,
                                                                     functor);
        }
    }
}

#pragma region Instantiate
#define InstantiateInvokeLutInplace_For(typeSrcIsTypeDst)                                                              \
    template void InvokeLutInplace<typeSrcIsTypeDst>(                                                                  \
        typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1, const Pixel32fC1 *const *aLevels,                           \
        const Pixel32fC1 *const *aValues, const int *const *aAccelerator, int const *aLutSize,                         \
        int const *aAcceleratorSize, InterpolationMode aInterpolationMode, const Size2D &aSize,                        \
        const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLutInplace(type)                                                                  \
    InstantiateInvokeLutInplace_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeLutInplace_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeLutInplace_For(Pixel##type##C4);                                                                  \
    InstantiateInvokeLutInplace_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeLutTrilinearSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst,
                           const Vector3<remove_vector_t<SrcDstT>> *aLut3D,
                           const Vector3<remove_vector_t<SrcDstT>> &aMinLevel,
                           const Vector3<remove_vector_t<SrcDstT>> &aMaxLevel, const Pixel32sC3 &aLutSize,
                           const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = SrcDstT;
    using LutT     = Vector3<remove_vector_t<SrcDstT>>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using lutSrc =
        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTTrilinear<SrcDstT, LutT>, RoundingMode::None>;

    const mpp::image::LUTTrilinear<SrcDstT, LutT> op(aLut3D, aMinLevel, aMaxLevel - aMinLevel, aLutSize);

    const lutSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLutTrilinearSrc3_For(typeSrcIsTypeDst)                                                        \
    template void InvokeLutTrilinearSrc<typeSrcIsTypeDst>(                                                             \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        const Vector3<remove_vector_t<typeSrcIsTypeDst>> *aLut3D,                                                      \
        const Vector3<remove_vector_t<typeSrcIsTypeDst>> &aMinLevel,                                                   \
        const Vector3<remove_vector_t<typeSrcIsTypeDst>> &aMaxLevel, const Pixel32sC3 &aLutSize, const Size2D &aSize,  \
        const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLutTrilinearSrc3(type)                                                            \
    InstantiateInvokeLutTrilinearSrc3_For(Pixel##type##C3);                                                            \
    InstantiateInvokeLutTrilinearSrc3_For(Pixel##type##C4);                                                            \
    InstantiateInvokeLutTrilinearSrc3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeLutTrilinearInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, const Vector3<remove_vector_t<SrcDstT>> *aLut3D,
                               const Vector3<remove_vector_t<SrcDstT>> &aMinLevel,
                               const Vector3<remove_vector_t<SrcDstT>> &aMaxLevel, const Pixel32sC3 &aLutSize,
                               const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = SrcDstT;
    using LutT     = Vector3<remove_vector_t<SrcDstT>>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using lutSrc =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::LUTTrilinear<SrcDstT, LutT>, RoundingMode::None>;

    const mpp::image::LUTTrilinear<SrcDstT, LutT> op(aLut3D, aMinLevel, aMaxLevel - aMinLevel, aLutSize);
    const lutSrc functor(op);
    InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLutTrilinearInplace3_For(typeSrcIsTypeDst)                                                    \
    template void InvokeLutTrilinearInplace<typeSrcIsTypeDst>(                                                         \
        typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1, const Vector3<remove_vector_t<typeSrcIsTypeDst>> *aLut3D,   \
        const Vector3<remove_vector_t<typeSrcIsTypeDst>> &aMinLevel,                                                   \
        const Vector3<remove_vector_t<typeSrcIsTypeDst>> &aMaxLevel, const Pixel32sC3 &aLutSize, const Size2D &aSize,  \
        const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLutTrilinearInplace3(type)                                                        \
    InstantiateInvokeLutTrilinearInplace3_For(Pixel##type##C3);                                                        \
    InstantiateInvokeLutTrilinearInplace3_For(Pixel##type##C4);                                                        \
    InstantiateInvokeLutTrilinearInplace3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeLutTrilinearSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst,
                           const Vector4A<remove_vector_t<SrcDstT>> *aLut3D,
                           const Vector3<remove_vector_t<SrcDstT>> &aMinLevel,
                           const Vector3<remove_vector_t<SrcDstT>> &aMaxLevel, const Pixel32sC3 &aLutSize,
                           const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = SrcDstT;
    using LutT     = Vector4A<remove_vector_t<SrcDstT>>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using lutSrc =
        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTTrilinear<SrcDstT, LutT>, RoundingMode::None>;

    const mpp::image::LUTTrilinear<SrcDstT, LutT> op(aLut3D, aMinLevel, aMaxLevel - aMinLevel, aLutSize);

    const lutSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLutTrilinearSrc4A_For(typeSrcIsTypeDst)                                                       \
    template void InvokeLutTrilinearSrc<typeSrcIsTypeDst>(                                                             \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        const Vector4A<remove_vector_t<typeSrcIsTypeDst>> *aLut3D,                                                     \
        const Vector3<remove_vector_t<typeSrcIsTypeDst>> &aMinLevel,                                                   \
        const Vector3<remove_vector_t<typeSrcIsTypeDst>> &aMaxLevel, const Pixel32sC3 &aLutSize, const Size2D &aSize,  \
        const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLutTrilinearSrc4A(type)                                                           \
    InstantiateInvokeLutTrilinearSrc4A_For(Pixel##type##C3);                                                           \
    InstantiateInvokeLutTrilinearSrc4A_For(Pixel##type##C4);                                                           \
    InstantiateInvokeLutTrilinearSrc4A_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeLutTrilinearInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1,
                               const Vector4A<remove_vector_t<SrcDstT>> *aLut3D,
                               const Vector3<remove_vector_t<SrcDstT>> &aMinLevel,
                               const Vector3<remove_vector_t<SrcDstT>> &aMaxLevel, const Pixel32sC3 &aLutSize,
                               const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = SrcDstT;
    using LutT     = Vector4A<remove_vector_t<SrcDstT>>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using lutSrc =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::LUTTrilinear<SrcDstT, LutT>, RoundingMode::None>;

    const mpp::image::LUTTrilinear<SrcDstT, LutT> op(aLut3D, aMinLevel, aMaxLevel - aMinLevel, aLutSize);
    const lutSrc functor(op);
    InvokeForEachPixelKernelDefault<DstT, TupelSize, lutSrc>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLutTrilinearInplace4A_For(typeSrcIsTypeDst)                                                   \
    template void InvokeLutTrilinearInplace<typeSrcIsTypeDst>(                                                         \
        typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1, const Vector4A<remove_vector_t<typeSrcIsTypeDst>> *aLut3D,  \
        const Vector3<remove_vector_t<typeSrcIsTypeDst>> &aMinLevel,                                                   \
        const Vector3<remove_vector_t<typeSrcIsTypeDst>> &aMaxLevel, const Pixel32sC3 &aLutSize, const Size2D &aSize,  \
        const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLutTrilinearInplace4A(type)                                                       \
    InstantiateInvokeLutTrilinearInplace4A_For(Pixel##type##C3);                                                       \
    InstantiateInvokeLutTrilinearInplace4A_For(Pixel##type##C4);                                                       \
    InstantiateInvokeLutTrilinearInplace4A_For(Pixel##type##C4A);

#pragma endregion
} // namespace mpp::image::cuda
