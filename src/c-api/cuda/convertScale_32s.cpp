// deactivate the DLL export macros in imageView:
#include <backends/cuda/image/dllexport_cudai.h>
#undef MPPEXPORT_CUDAI
#define MPPEXPORT_CUDAI
#define ENABLE_CUDA_HALFFLOAT
#define ENABLE_CUDA_BFLOAT
// NOLINTBEGIN(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp,modernize-use-using,cppcoreguidelines-pro-type-const-cast,misc-include-cleaner)
#include "catch_and_return.h"
#include "dllexport.h"
#include "mppc_capi_defs.h"
#include <algorithm>
#include <backends/cuda/image/imageView.h>
#include <backends/cuda/image/imageView_arithmetic_impl.h>          //NOLINT(misc-include-cleaner)
#include <backends/cuda/image/imageView_colorConversion_impl.h>     //NOLINT(misc-include-cleaner)
#include <backends/cuda/image/imageView_dataExchangeAndInit_impl.h> //NOLINT(misc-include-cleaner)
#include <backends/cuda/image/imageView_filtering_impl.h>           //NOLINT(misc-include-cleaner)
#include <backends/cuda/image/imageView_geometryTransforms_impl.h>  //NOLINT(misc-include-cleaner)
#include <backends/cuda/image/imageView_morphology_impl.h>          //NOLINT(misc-include-cleaner)
#include <backends/cuda/image/imageView_statistics_impl.h>          //NOLINT(misc-include-cleaner)
#include <backends/cuda/image/imageView_thresholdAndCompare_impl.h> //NOLINT(misc-include-cleaner)
#include <backends/cuda/streamCtx.h>                                //NOLINT(misc-include-cleaner)
#include <common/errorMessageSingleton.h>
#include <common/exception.h>
#include <common/image/bound.h>
#include <common/image/matrix.h>
#include <common/image/pixelTypes.h>
#include <common/mpp_defs.h>
#include <common/vectorTypes.h>
#include <exception>

using namespace mpp;
using namespace mpp::image;
using namespace mpp::cuda;
using namespace mpp::image::cuda;

extern "C"
{
    MPPErrorCode DLLEXPORT mppciConvert_32s8s_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                                 size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC1> _Dst(reinterpret_cast<Pixel8sC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s8s_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                                 size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC2> _Dst(reinterpret_cast<Pixel8sC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s8s_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                                 size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC3> _Dst(reinterpret_cast<Pixel8sC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s8s_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                                 size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC4> _Dst(reinterpret_cast<Pixel8sC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s8s_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC4A> _Dst(reinterpret_cast<Pixel8sC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciConvert_32s8s_C1Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                                    size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                    int aScaleFactor, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC1> _Dst(reinterpret_cast<Pixel8sC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s8s_C2Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                                    size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                    int aScaleFactor, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC2> _Dst(reinterpret_cast<Pixel8sC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s8s_C3Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                                    size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                    int aScaleFactor, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC3> _Dst(reinterpret_cast<Pixel8sC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s8s_C4Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                                    size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                    int aScaleFactor, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC4> _Dst(reinterpret_cast<Pixel8sC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s8s_AC4Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                                     size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     int aScaleFactor, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC4A> _Dst(reinterpret_cast<Pixel8sC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciScale_32s8s_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                               Mpp32s aSrcMax, DevPtrMpp8s aDst, size_t aDstStep, Mpp8s aDstMin,
                                               Mpp8s aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                               CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC1> _Dst(reinterpret_cast<Pixel8sC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s8s_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                               Mpp32s aSrcMax, DevPtrMpp8s aDst, size_t aDstStep, Mpp8s aDstMin,
                                               Mpp8s aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                               CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC2> _Dst(reinterpret_cast<Pixel8sC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s8s_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                               Mpp32s aSrcMax, DevPtrMpp8s aDst, size_t aDstStep, Mpp8s aDstMin,
                                               Mpp8s aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                               CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC3> _Dst(reinterpret_cast<Pixel8sC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s8s_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                               Mpp32s aSrcMax, DevPtrMpp8s aDst, size_t aDstStep, Mpp8s aDstMin,
                                               Mpp8s aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                               CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC4> _Dst(reinterpret_cast<Pixel8sC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s8s_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp8s aDst, size_t aDstStep, Mpp8s aDstMin,
                                                Mpp8s aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC4A> _Dst(reinterpret_cast<Pixel8sC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s8s_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                                  size_t aDstStep, Mpp8s aDstMin, Mpp8s aDstMax, MppiSize aSizeROI,
                                                  MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC1> _Dst(reinterpret_cast<Pixel8sC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s8s_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                                  size_t aDstStep, Mpp8s aDstMin, Mpp8s aDstMax, MppiSize aSizeROI,
                                                  MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC2> _Dst(reinterpret_cast<Pixel8sC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s8s_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                                  size_t aDstStep, Mpp8s aDstMin, Mpp8s aDstMax, MppiSize aSizeROI,
                                                  MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC3> _Dst(reinterpret_cast<Pixel8sC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s8s_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                                  size_t aDstStep, Mpp8s aDstMin, Mpp8s aDstMax, MppiSize aSizeROI,
                                                  MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC4> _Dst(reinterpret_cast<Pixel8sC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s8s_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                                   size_t aDstStep, Mpp8s aDstMin, Mpp8s aDstMax, MppiSize aSizeROI,
                                                   MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC4A> _Dst(reinterpret_cast<Pixel8sC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s8s_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                                       size_t aDstStep, MppiSize aSizeROI,
                                                       MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC1> _Dst(reinterpret_cast<Pixel8sC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s8s_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                                       size_t aDstStep, MppiSize aSizeROI,
                                                       MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC2> _Dst(reinterpret_cast<Pixel8sC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s8s_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                                       size_t aDstStep, MppiSize aSizeROI,
                                                       MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC3> _Dst(reinterpret_cast<Pixel8sC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s8s_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                                       size_t aDstStep, MppiSize aSizeROI,
                                                       MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC4> _Dst(reinterpret_cast<Pixel8sC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s8s_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                                        size_t aDstStep, MppiSize aSizeROI,
                                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC4A> _Dst(reinterpret_cast<Pixel8sC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s8s_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                    Mpp32s aSrcMax, DevPtrMpp8s aDst, size_t aDstStep,
                                                    MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                    CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC1> _Dst(reinterpret_cast<Pixel8sC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s8s_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                    Mpp32s aSrcMax, DevPtrMpp8s aDst, size_t aDstStep,
                                                    MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                    CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC2> _Dst(reinterpret_cast<Pixel8sC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s8s_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                    Mpp32s aSrcMax, DevPtrMpp8s aDst, size_t aDstStep,
                                                    MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                    CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC3> _Dst(reinterpret_cast<Pixel8sC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s8s_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                    Mpp32s aSrcMax, DevPtrMpp8s aDst, size_t aDstStep,
                                                    MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                    CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC4> _Dst(reinterpret_cast<Pixel8sC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s8s_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                     Mpp32s aSrcMax, DevPtrMpp8s aDst, size_t aDstStep,
                                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel8sC4A> _Dst(reinterpret_cast<Pixel8sC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s8u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                                 size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC1> _Dst(reinterpret_cast<Pixel8uC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s8u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                                 size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC2> _Dst(reinterpret_cast<Pixel8uC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s8u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                                 size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC3> _Dst(reinterpret_cast<Pixel8uC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s8u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                                 size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC4> _Dst(reinterpret_cast<Pixel8uC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s8u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC4A> _Dst(reinterpret_cast<Pixel8uC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciConvert_32s8u_C1Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                                    size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                    int aScaleFactor, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC1> _Dst(reinterpret_cast<Pixel8uC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s8u_C2Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                                    size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                    int aScaleFactor, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC2> _Dst(reinterpret_cast<Pixel8uC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s8u_C3Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                                    size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                    int aScaleFactor, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC3> _Dst(reinterpret_cast<Pixel8uC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s8u_C4Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                                    size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                    int aScaleFactor, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC4> _Dst(reinterpret_cast<Pixel8uC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s8u_AC4Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                                     size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     int aScaleFactor, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC4A> _Dst(reinterpret_cast<Pixel8uC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciScale_32s8u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                               Mpp32s aSrcMax, DevPtrMpp8u aDst, size_t aDstStep, Mpp8u aDstMin,
                                               Mpp8u aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                               CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC1> _Dst(reinterpret_cast<Pixel8uC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s8u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                               Mpp32s aSrcMax, DevPtrMpp8u aDst, size_t aDstStep, Mpp8u aDstMin,
                                               Mpp8u aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                               CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC2> _Dst(reinterpret_cast<Pixel8uC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s8u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                               Mpp32s aSrcMax, DevPtrMpp8u aDst, size_t aDstStep, Mpp8u aDstMin,
                                               Mpp8u aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                               CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC3> _Dst(reinterpret_cast<Pixel8uC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s8u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                               Mpp32s aSrcMax, DevPtrMpp8u aDst, size_t aDstStep, Mpp8u aDstMin,
                                               Mpp8u aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                               CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC4> _Dst(reinterpret_cast<Pixel8uC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s8u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp8u aDst, size_t aDstStep, Mpp8u aDstMin,
                                                Mpp8u aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC4A> _Dst(reinterpret_cast<Pixel8uC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s8u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                                  size_t aDstStep, Mpp8u aDstMin, Mpp8u aDstMax, MppiSize aSizeROI,
                                                  MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC1> _Dst(reinterpret_cast<Pixel8uC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s8u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                                  size_t aDstStep, Mpp8u aDstMin, Mpp8u aDstMax, MppiSize aSizeROI,
                                                  MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC2> _Dst(reinterpret_cast<Pixel8uC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s8u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                                  size_t aDstStep, Mpp8u aDstMin, Mpp8u aDstMax, MppiSize aSizeROI,
                                                  MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC3> _Dst(reinterpret_cast<Pixel8uC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s8u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                                  size_t aDstStep, Mpp8u aDstMin, Mpp8u aDstMax, MppiSize aSizeROI,
                                                  MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC4> _Dst(reinterpret_cast<Pixel8uC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s8u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                                   size_t aDstStep, Mpp8u aDstMin, Mpp8u aDstMax, MppiSize aSizeROI,
                                                   MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC4A> _Dst(reinterpret_cast<Pixel8uC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s8u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                                       size_t aDstStep, MppiSize aSizeROI,
                                                       MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC1> _Dst(reinterpret_cast<Pixel8uC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s8u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                                       size_t aDstStep, MppiSize aSizeROI,
                                                       MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC2> _Dst(reinterpret_cast<Pixel8uC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s8u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                                       size_t aDstStep, MppiSize aSizeROI,
                                                       MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC3> _Dst(reinterpret_cast<Pixel8uC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s8u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                                       size_t aDstStep, MppiSize aSizeROI,
                                                       MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC4> _Dst(reinterpret_cast<Pixel8uC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s8u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                                        size_t aDstStep, MppiSize aSizeROI,
                                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC4A> _Dst(reinterpret_cast<Pixel8uC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s8u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                    Mpp32s aSrcMax, DevPtrMpp8u aDst, size_t aDstStep,
                                                    MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                    CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC1> _Dst(reinterpret_cast<Pixel8uC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s8u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                    Mpp32s aSrcMax, DevPtrMpp8u aDst, size_t aDstStep,
                                                    MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                    CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC2> _Dst(reinterpret_cast<Pixel8uC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s8u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                    Mpp32s aSrcMax, DevPtrMpp8u aDst, size_t aDstStep,
                                                    MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                    CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC3> _Dst(reinterpret_cast<Pixel8uC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s8u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                    Mpp32s aSrcMax, DevPtrMpp8u aDst, size_t aDstStep,
                                                    MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                    CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC4> _Dst(reinterpret_cast<Pixel8uC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s8u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                     Mpp32s aSrcMax, DevPtrMpp8u aDst, size_t aDstStep,
                                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel8uC4A> _Dst(reinterpret_cast<Pixel8uC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16s_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC1> _Dst(reinterpret_cast<Pixel16sC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16s_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC2> _Dst(reinterpret_cast<Pixel16sC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16s_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC3> _Dst(reinterpret_cast<Pixel16sC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16s_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC4> _Dst(reinterpret_cast<Pixel16sC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16s_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                                   size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC4A> _Dst(reinterpret_cast<Pixel16sC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciConvert_32s16s_C1Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                                     size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     int aScaleFactor, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC1> _Dst(reinterpret_cast<Pixel16sC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16s_C2Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                                     size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     int aScaleFactor, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC2> _Dst(reinterpret_cast<Pixel16sC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16s_C3Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                                     size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     int aScaleFactor, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC3> _Dst(reinterpret_cast<Pixel16sC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16s_C4Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                                     size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     int aScaleFactor, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC4> _Dst(reinterpret_cast<Pixel16sC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16s_AC4Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                                      size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                      int aScaleFactor, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC4A> _Dst(reinterpret_cast<Pixel16sC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciScale_32s16s_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp16s aDst, size_t aDstStep, Mpp16s aDstMin,
                                                Mpp16s aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC1> _Dst(reinterpret_cast<Pixel16sC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s16s_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp16s aDst, size_t aDstStep, Mpp16s aDstMin,
                                                Mpp16s aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC2> _Dst(reinterpret_cast<Pixel16sC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s16s_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp16s aDst, size_t aDstStep, Mpp16s aDstMin,
                                                Mpp16s aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC3> _Dst(reinterpret_cast<Pixel16sC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s16s_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp16s aDst, size_t aDstStep, Mpp16s aDstMin,
                                                Mpp16s aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC4> _Dst(reinterpret_cast<Pixel16sC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s16s_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                 Mpp32s aSrcMax, DevPtrMpp16s aDst, size_t aDstStep, Mpp16s aDstMin,
                                                 Mpp16s aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                 CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC4A> _Dst(reinterpret_cast<Pixel16sC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s16s_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                                   size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax, MppiSize aSizeROI,
                                                   MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC1> _Dst(reinterpret_cast<Pixel16sC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s16s_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                                   size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax, MppiSize aSizeROI,
                                                   MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC2> _Dst(reinterpret_cast<Pixel16sC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s16s_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                                   size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax, MppiSize aSizeROI,
                                                   MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC3> _Dst(reinterpret_cast<Pixel16sC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s16s_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                                   size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax, MppiSize aSizeROI,
                                                   MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC4> _Dst(reinterpret_cast<Pixel16sC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s16s_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                                    size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax, MppiSize aSizeROI,
                                                    MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC4A> _Dst(reinterpret_cast<Pixel16sC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s16s_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                                        size_t aDstStep, MppiSize aSizeROI,
                                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC1> _Dst(reinterpret_cast<Pixel16sC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s16s_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                                        size_t aDstStep, MppiSize aSizeROI,
                                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC2> _Dst(reinterpret_cast<Pixel16sC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s16s_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                                        size_t aDstStep, MppiSize aSizeROI,
                                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC3> _Dst(reinterpret_cast<Pixel16sC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s16s_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                                        size_t aDstStep, MppiSize aSizeROI,
                                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC4> _Dst(reinterpret_cast<Pixel16sC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s16s_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                                         size_t aDstStep, MppiSize aSizeROI,
                                                         MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC4A> _Dst(reinterpret_cast<Pixel16sC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s16s_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                     Mpp32s aSrcMax, DevPtrMpp16s aDst, size_t aDstStep,
                                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC1> _Dst(reinterpret_cast<Pixel16sC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s16s_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                     Mpp32s aSrcMax, DevPtrMpp16s aDst, size_t aDstStep,
                                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC2> _Dst(reinterpret_cast<Pixel16sC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s16s_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                     Mpp32s aSrcMax, DevPtrMpp16s aDst, size_t aDstStep,
                                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC3> _Dst(reinterpret_cast<Pixel16sC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s16s_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                     Mpp32s aSrcMax, DevPtrMpp16s aDst, size_t aDstStep,
                                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC4> _Dst(reinterpret_cast<Pixel16sC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s16s_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                      Mpp32s aSrcMax, DevPtrMpp16s aDst, size_t aDstStep,
                                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                      CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16sC4A> _Dst(reinterpret_cast<Pixel16sC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC1> _Dst(reinterpret_cast<Pixel16uC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC2> _Dst(reinterpret_cast<Pixel16uC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC3> _Dst(reinterpret_cast<Pixel16uC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC4> _Dst(reinterpret_cast<Pixel16uC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                                   size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC4A> _Dst(reinterpret_cast<Pixel16uC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciConvert_32s16u_C1Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                                     size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     int aScaleFactor, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC1> _Dst(reinterpret_cast<Pixel16uC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16u_C2Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                                     size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     int aScaleFactor, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC2> _Dst(reinterpret_cast<Pixel16uC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16u_C3Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                                     size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     int aScaleFactor, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC3> _Dst(reinterpret_cast<Pixel16uC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16u_C4Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                                     size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     int aScaleFactor, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC4> _Dst(reinterpret_cast<Pixel16uC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16u_AC4Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                                      size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                      int aScaleFactor, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC4A> _Dst(reinterpret_cast<Pixel16uC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciScale_32s16u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp16u aDst, size_t aDstStep, Mpp16u aDstMin,
                                                Mpp16u aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC1> _Dst(reinterpret_cast<Pixel16uC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s16u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp16u aDst, size_t aDstStep, Mpp16u aDstMin,
                                                Mpp16u aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC2> _Dst(reinterpret_cast<Pixel16uC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s16u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp16u aDst, size_t aDstStep, Mpp16u aDstMin,
                                                Mpp16u aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC3> _Dst(reinterpret_cast<Pixel16uC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s16u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp16u aDst, size_t aDstStep, Mpp16u aDstMin,
                                                Mpp16u aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC4> _Dst(reinterpret_cast<Pixel16uC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s16u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                 Mpp32s aSrcMax, DevPtrMpp16u aDst, size_t aDstStep, Mpp16u aDstMin,
                                                 Mpp16u aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                 CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC4A> _Dst(reinterpret_cast<Pixel16uC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s16u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                                   size_t aDstStep, Mpp16u aDstMin, Mpp16u aDstMax, MppiSize aSizeROI,
                                                   MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC1> _Dst(reinterpret_cast<Pixel16uC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s16u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                                   size_t aDstStep, Mpp16u aDstMin, Mpp16u aDstMax, MppiSize aSizeROI,
                                                   MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC2> _Dst(reinterpret_cast<Pixel16uC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s16u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                                   size_t aDstStep, Mpp16u aDstMin, Mpp16u aDstMax, MppiSize aSizeROI,
                                                   MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC3> _Dst(reinterpret_cast<Pixel16uC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s16u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                                   size_t aDstStep, Mpp16u aDstMin, Mpp16u aDstMax, MppiSize aSizeROI,
                                                   MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC4> _Dst(reinterpret_cast<Pixel16uC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s16u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                                    size_t aDstStep, Mpp16u aDstMin, Mpp16u aDstMax, MppiSize aSizeROI,
                                                    MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC4A> _Dst(reinterpret_cast<Pixel16uC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s16u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                                        size_t aDstStep, MppiSize aSizeROI,
                                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC1> _Dst(reinterpret_cast<Pixel16uC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s16u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                                        size_t aDstStep, MppiSize aSizeROI,
                                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC2> _Dst(reinterpret_cast<Pixel16uC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s16u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                                        size_t aDstStep, MppiSize aSizeROI,
                                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC3> _Dst(reinterpret_cast<Pixel16uC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s16u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                                        size_t aDstStep, MppiSize aSizeROI,
                                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC4> _Dst(reinterpret_cast<Pixel16uC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s16u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                                         size_t aDstStep, MppiSize aSizeROI,
                                                         MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC4A> _Dst(reinterpret_cast<Pixel16uC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s16u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                     Mpp32s aSrcMax, DevPtrMpp16u aDst, size_t aDstStep,
                                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC1> _Dst(reinterpret_cast<Pixel16uC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s16u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                     Mpp32s aSrcMax, DevPtrMpp16u aDst, size_t aDstStep,
                                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC2> _Dst(reinterpret_cast<Pixel16uC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s16u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                     Mpp32s aSrcMax, DevPtrMpp16u aDst, size_t aDstStep,
                                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC3> _Dst(reinterpret_cast<Pixel16uC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s16u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                     Mpp32s aSrcMax, DevPtrMpp16u aDst, size_t aDstStep,
                                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC4> _Dst(reinterpret_cast<Pixel16uC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s16u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                      Mpp32s aSrcMax, DevPtrMpp16u aDst, size_t aDstStep,
                                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                      CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16uC4A> _Dst(reinterpret_cast<Pixel16uC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s32u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC1> _Dst(reinterpret_cast<Pixel32uC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s32u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC2> _Dst(reinterpret_cast<Pixel32uC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s32u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC3> _Dst(reinterpret_cast<Pixel32uC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s32u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC4> _Dst(reinterpret_cast<Pixel32uC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s32u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                                   size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC4A> _Dst(reinterpret_cast<Pixel32uC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciScale_32s32u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp32u aDst, size_t aDstStep, Mpp32u aDstMin,
                                                Mpp32u aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC1> _Dst(reinterpret_cast<Pixel32uC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s32u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp32u aDst, size_t aDstStep, Mpp32u aDstMin,
                                                Mpp32u aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC2> _Dst(reinterpret_cast<Pixel32uC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s32u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp32u aDst, size_t aDstStep, Mpp32u aDstMin,
                                                Mpp32u aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC3> _Dst(reinterpret_cast<Pixel32uC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s32u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp32u aDst, size_t aDstStep, Mpp32u aDstMin,
                                                Mpp32u aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC4> _Dst(reinterpret_cast<Pixel32uC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s32u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                 Mpp32s aSrcMax, DevPtrMpp32u aDst, size_t aDstStep, Mpp32u aDstMin,
                                                 Mpp32u aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                 CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC4A> _Dst(reinterpret_cast<Pixel32uC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s32u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                                   size_t aDstStep, Mpp32u aDstMin, Mpp32u aDstMax, MppiSize aSizeROI,
                                                   MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC1> _Dst(reinterpret_cast<Pixel32uC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s32u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                                   size_t aDstStep, Mpp32u aDstMin, Mpp32u aDstMax, MppiSize aSizeROI,
                                                   MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC2> _Dst(reinterpret_cast<Pixel32uC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s32u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                                   size_t aDstStep, Mpp32u aDstMin, Mpp32u aDstMax, MppiSize aSizeROI,
                                                   MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC3> _Dst(reinterpret_cast<Pixel32uC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s32u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                                   size_t aDstStep, Mpp32u aDstMin, Mpp32u aDstMax, MppiSize aSizeROI,
                                                   MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC4> _Dst(reinterpret_cast<Pixel32uC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s32u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                                    size_t aDstStep, Mpp32u aDstMin, Mpp32u aDstMax, MppiSize aSizeROI,
                                                    MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC4A> _Dst(reinterpret_cast<Pixel32uC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s32u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                                        size_t aDstStep, MppiSize aSizeROI,
                                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC1> _Dst(reinterpret_cast<Pixel32uC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s32u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                                        size_t aDstStep, MppiSize aSizeROI,
                                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC2> _Dst(reinterpret_cast<Pixel32uC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s32u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                                        size_t aDstStep, MppiSize aSizeROI,
                                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC3> _Dst(reinterpret_cast<Pixel32uC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s32u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                                        size_t aDstStep, MppiSize aSizeROI,
                                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC4> _Dst(reinterpret_cast<Pixel32uC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32s32u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                                         size_t aDstStep, MppiSize aSizeROI,
                                                         MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC4A> _Dst(reinterpret_cast<Pixel32uC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s32u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                     Mpp32s aSrcMax, DevPtrMpp32u aDst, size_t aDstStep,
                                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC1> _Dst(reinterpret_cast<Pixel32uC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s32u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                     Mpp32s aSrcMax, DevPtrMpp32u aDst, size_t aDstStep,
                                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC2> _Dst(reinterpret_cast<Pixel32uC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s32u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                     Mpp32s aSrcMax, DevPtrMpp32u aDst, size_t aDstStep,
                                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC3> _Dst(reinterpret_cast<Pixel32uC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s32u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                     Mpp32s aSrcMax, DevPtrMpp32u aDst, size_t aDstStep,
                                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                     CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC4> _Dst(reinterpret_cast<Pixel32uC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32s32u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                      Mpp32s aSrcMax, DevPtrMpp32u aDst, size_t aDstStep,
                                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                      CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel32uC4A> _Dst(reinterpret_cast<Pixel32uC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16f_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16fC1> _Dst(reinterpret_cast<Pixel16fC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16f_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16fC2> _Dst(reinterpret_cast<Pixel16fC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16f_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16fC3> _Dst(reinterpret_cast<Pixel16fC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16f_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16fC4> _Dst(reinterpret_cast<Pixel16fC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16f_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst,
                                                   size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16fC4A> _Dst(reinterpret_cast<Pixel16fC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciScale_32s16f_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp16f aDst, size_t aDstStep, Mpp16f aDstMin,
                                                Mpp16f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16fC1> _Dst(reinterpret_cast<Pixel16fC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s16f_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp16f aDst, size_t aDstStep, Mpp16f aDstMin,
                                                Mpp16f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16fC2> _Dst(reinterpret_cast<Pixel16fC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s16f_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp16f aDst, size_t aDstStep, Mpp16f aDstMin,
                                                Mpp16f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16fC3> _Dst(reinterpret_cast<Pixel16fC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s16f_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp16f aDst, size_t aDstStep, Mpp16f aDstMin,
                                                Mpp16f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16fC4> _Dst(reinterpret_cast<Pixel16fC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s16f_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                 Mpp32s aSrcMax, DevPtrMpp16f aDst, size_t aDstStep, Mpp16f aDstMin,
                                                 Mpp16f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16fC4A> _Dst(reinterpret_cast<Pixel16fC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s16f_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst,
                                                   size_t aDstStep, Mpp16f aDstMin, Mpp16f aDstMax, MppiSize aSizeROI,
                                                   CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16fC1> _Dst(reinterpret_cast<Pixel16fC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s16f_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst,
                                                   size_t aDstStep, Mpp16f aDstMin, Mpp16f aDstMax, MppiSize aSizeROI,
                                                   CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16fC2> _Dst(reinterpret_cast<Pixel16fC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s16f_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst,
                                                   size_t aDstStep, Mpp16f aDstMin, Mpp16f aDstMax, MppiSize aSizeROI,
                                                   CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16fC3> _Dst(reinterpret_cast<Pixel16fC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s16f_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst,
                                                   size_t aDstStep, Mpp16f aDstMin, Mpp16f aDstMax, MppiSize aSizeROI,
                                                   CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16fC4> _Dst(reinterpret_cast<Pixel16fC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s16f_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst,
                                                    size_t aDstStep, Mpp16f aDstMin, Mpp16f aDstMax, MppiSize aSizeROI,
                                                    CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16fC4A> _Dst(reinterpret_cast<Pixel16fC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16bf_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                                   size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC1> _Dst(reinterpret_cast<Pixel16bfC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16bf_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                                   size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC2> _Dst(reinterpret_cast<Pixel16bfC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16bf_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                                   size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC3> _Dst(reinterpret_cast<Pixel16bfC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16bf_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                                   size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC4> _Dst(reinterpret_cast<Pixel16bfC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s16bf_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                                    size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC4A> _Dst(reinterpret_cast<Pixel16bfC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciScale_32s16bf_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                 Mpp32s aSrcMax, DevPtrMpp16bf aDst, size_t aDstStep, Mpp16bf aDstMin,
                                                 Mpp16bf aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC1> _Dst(reinterpret_cast<Pixel16bfC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s16bf_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                 Mpp32s aSrcMax, DevPtrMpp16bf aDst, size_t aDstStep, Mpp16bf aDstMin,
                                                 Mpp16bf aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC2> _Dst(reinterpret_cast<Pixel16bfC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s16bf_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                 Mpp32s aSrcMax, DevPtrMpp16bf aDst, size_t aDstStep, Mpp16bf aDstMin,
                                                 Mpp16bf aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC3> _Dst(reinterpret_cast<Pixel16bfC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s16bf_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                 Mpp32s aSrcMax, DevPtrMpp16bf aDst, size_t aDstStep, Mpp16bf aDstMin,
                                                 Mpp16bf aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC4> _Dst(reinterpret_cast<Pixel16bfC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s16bf_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                  Mpp32s aSrcMax, DevPtrMpp16bf aDst, size_t aDstStep, Mpp16bf aDstMin,
                                                  Mpp16bf aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC4A> _Dst(reinterpret_cast<Pixel16bfC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s16bf_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                                    size_t aDstStep, Mpp16bf aDstMin, Mpp16bf aDstMax,
                                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC1> _Dst(reinterpret_cast<Pixel16bfC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s16bf_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                                    size_t aDstStep, Mpp16bf aDstMin, Mpp16bf aDstMax,
                                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC2> _Dst(reinterpret_cast<Pixel16bfC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s16bf_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                                    size_t aDstStep, Mpp16bf aDstMin, Mpp16bf aDstMax,
                                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC3> _Dst(reinterpret_cast<Pixel16bfC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s16bf_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                                    size_t aDstStep, Mpp16bf aDstMin, Mpp16bf aDstMax,
                                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC4> _Dst(reinterpret_cast<Pixel16bfC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s16bf_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                                     size_t aDstStep, Mpp16bf aDstMin, Mpp16bf aDstMax,
                                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC4A> _Dst(reinterpret_cast<Pixel16bfC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s32f_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32fC1> _Dst(reinterpret_cast<Pixel32fC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s32f_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32fC2> _Dst(reinterpret_cast<Pixel32fC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s32f_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32fC3> _Dst(reinterpret_cast<Pixel32fC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s32f_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32fC4> _Dst(reinterpret_cast<Pixel32fC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s32f_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst,
                                                   size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel32fC4A> _Dst(reinterpret_cast<Pixel32fC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciScale_32s32f_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp32f aDst, size_t aDstStep, Mpp32f aDstMin,
                                                Mpp32f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32fC1> _Dst(reinterpret_cast<Pixel32fC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s32f_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp32f aDst, size_t aDstStep, Mpp32f aDstMin,
                                                Mpp32f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32fC2> _Dst(reinterpret_cast<Pixel32fC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s32f_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp32f aDst, size_t aDstStep, Mpp32f aDstMin,
                                                Mpp32f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32fC3> _Dst(reinterpret_cast<Pixel32fC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s32f_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp32f aDst, size_t aDstStep, Mpp32f aDstMin,
                                                Mpp32f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32fC4> _Dst(reinterpret_cast<Pixel32fC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s32f_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                 Mpp32s aSrcMax, DevPtrMpp32f aDst, size_t aDstStep, Mpp32f aDstMin,
                                                 Mpp32f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel32fC4A> _Dst(reinterpret_cast<Pixel32fC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s32f_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst,
                                                   size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax, MppiSize aSizeROI,
                                                   CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32fC1> _Dst(reinterpret_cast<Pixel32fC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s32f_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst,
                                                   size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax, MppiSize aSizeROI,
                                                   CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32fC2> _Dst(reinterpret_cast<Pixel32fC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s32f_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst,
                                                   size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax, MppiSize aSizeROI,
                                                   CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32fC3> _Dst(reinterpret_cast<Pixel32fC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s32f_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst,
                                                   size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax, MppiSize aSizeROI,
                                                   CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel32fC4> _Dst(reinterpret_cast<Pixel32fC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s32f_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst,
                                                    size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax, MppiSize aSizeROI,
                                                    CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel32fC4A> _Dst(reinterpret_cast<Pixel32fC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s64f_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel64fC1> _Dst(reinterpret_cast<Pixel64fC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s64f_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel64fC2> _Dst(reinterpret_cast<Pixel64fC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s64f_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel64fC3> _Dst(reinterpret_cast<Pixel64fC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s64f_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst,
                                                  size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel64fC4> _Dst(reinterpret_cast<Pixel64fC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32s64f_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst,
                                                   size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel64fC4A> _Dst(reinterpret_cast<Pixel64fC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciScale_32s64f_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp64f aDst, size_t aDstStep, Mpp64f aDstMin,
                                                Mpp64f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel64fC1> _Dst(reinterpret_cast<Pixel64fC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s64f_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp64f aDst, size_t aDstStep, Mpp64f aDstMin,
                                                Mpp64f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel64fC2> _Dst(reinterpret_cast<Pixel64fC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s64f_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp64f aDst, size_t aDstStep, Mpp64f aDstMin,
                                                Mpp64f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel64fC3> _Dst(reinterpret_cast<Pixel64fC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s64f_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                Mpp32s aSrcMax, DevPtrMpp64f aDst, size_t aDstStep, Mpp64f aDstMin,
                                                Mpp64f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel64fC4> _Dst(reinterpret_cast<Pixel64fC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32s64f_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                 Mpp32s aSrcMax, DevPtrMpp64f aDst, size_t aDstStep, Mpp64f aDstMin,
                                                 Mpp64f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel64fC4A> _Dst(reinterpret_cast<Pixel64fC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s64f_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst,
                                                   size_t aDstStep, Mpp64f aDstMin, Mpp64f aDstMax, MppiSize aSizeROI,
                                                   CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC1> _Src1(reinterpret_cast<Pixel32sC1 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel64fC1> _Dst(reinterpret_cast<Pixel64fC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s64f_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst,
                                                   size_t aDstStep, Mpp64f aDstMin, Mpp64f aDstMax, MppiSize aSizeROI,
                                                   CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC2> _Src1(reinterpret_cast<Pixel32sC2 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel64fC2> _Dst(reinterpret_cast<Pixel64fC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s64f_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst,
                                                   size_t aDstStep, Mpp64f aDstMin, Mpp64f aDstMax, MppiSize aSizeROI,
                                                   CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC3> _Src1(reinterpret_cast<Pixel32sC3 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel64fC3> _Dst(reinterpret_cast<Pixel64fC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s64f_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst,
                                                   size_t aDstStep, Mpp64f aDstMin, Mpp64f aDstMax, MppiSize aSizeROI,
                                                   CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4> _Src1(reinterpret_cast<Pixel32sC4 *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                              {_SizeROI, aSrc1Step});
            ImageView<Pixel64fC4> _Dst(reinterpret_cast<Pixel64fC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32s64f_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst,
                                                    size_t aDstStep, Mpp64f aDstMin, Mpp64f aDstMax, MppiSize aSizeROI,
                                                    CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32sC4A> _Src1(reinterpret_cast<Pixel32sC4A *>(const_cast<DevPtrMpp32s>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel64fC4A> _Dst(reinterpret_cast<Pixel64fC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
} // extern "C"
// NOLINTEND(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp,modernize-use-using,cppcoreguidelines-pro-type-const-cast,misc-include-cleaner)
