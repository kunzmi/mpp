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
    MPPErrorCode DLLEXPORT mppciConvert_32sc16sc_C1(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                                    size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC1> _Src1(reinterpret_cast<Pixel32scC1 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC1> _Dst(reinterpret_cast<Pixel16scC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32sc16sc_C2(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                                    size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC2> _Src1(reinterpret_cast<Pixel32scC2 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC2> _Dst(reinterpret_cast<Pixel16scC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32sc16sc_C3(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                                    size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC3> _Src1(reinterpret_cast<Pixel32scC3 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC3> _Dst(reinterpret_cast<Pixel16scC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32sc16sc_C4(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                                    size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC4> _Src1(reinterpret_cast<Pixel32scC4 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC4> _Dst(reinterpret_cast<Pixel16scC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciConvert_32sc16sc_C1Sfs(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                                       size_t aDstStep, MppiSize aSizeROI,
                                                       MPPRoundingMode aRoundingMode, int aScaleFactor,
                                                       CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC1> _Src1(reinterpret_cast<Pixel32scC1 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC1> _Dst(reinterpret_cast<Pixel16scC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32sc16sc_C2Sfs(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                                       size_t aDstStep, MppiSize aSizeROI,
                                                       MPPRoundingMode aRoundingMode, int aScaleFactor,
                                                       CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC2> _Src1(reinterpret_cast<Pixel32scC2 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC2> _Dst(reinterpret_cast<Pixel16scC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32sc16sc_C3Sfs(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                                       size_t aDstStep, MppiSize aSizeROI,
                                                       MPPRoundingMode aRoundingMode, int aScaleFactor,
                                                       CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC3> _Src1(reinterpret_cast<Pixel32scC3 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC3> _Dst(reinterpret_cast<Pixel16scC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32sc16sc_C4Sfs(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                                       size_t aDstStep, MppiSize aSizeROI,
                                                       MPPRoundingMode aRoundingMode, int aScaleFactor,
                                                       CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC4> _Src1(reinterpret_cast<Pixel32scC4 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC4> _Dst(reinterpret_cast<Pixel16scC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, static_cast<RoundingMode>(aRoundingMode), aScaleFactor, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciScale_32sc16sc_C1(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                  Mpp32s aSrcMax, DevPtrMpp16sc aDst, size_t aDstStep, Mpp16s aDstMin,
                                                  Mpp16s aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                  CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC1> _Src1(reinterpret_cast<Pixel32scC1 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC1> _Dst(reinterpret_cast<Pixel16scC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32sc16sc_C2(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                  Mpp32s aSrcMax, DevPtrMpp16sc aDst, size_t aDstStep, Mpp16s aDstMin,
                                                  Mpp16s aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                  CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC2> _Src1(reinterpret_cast<Pixel32scC2 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC2> _Dst(reinterpret_cast<Pixel16scC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32sc16sc_C3(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                  Mpp32s aSrcMax, DevPtrMpp16sc aDst, size_t aDstStep, Mpp16s aDstMin,
                                                  Mpp16s aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                  CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC3> _Src1(reinterpret_cast<Pixel32scC3 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC3> _Dst(reinterpret_cast<Pixel16scC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32sc16sc_C4(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                  Mpp32s aSrcMax, DevPtrMpp16sc aDst, size_t aDstStep, Mpp16s aDstMin,
                                                  Mpp16s aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                  CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC4> _Src1(reinterpret_cast<Pixel32scC4 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC4> _Dst(reinterpret_cast<Pixel16scC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode),
                        *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32sc16sc_C1(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                                     size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax, MppiSize aSizeROI,
                                                     MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC1> _Src1(reinterpret_cast<Pixel32scC1 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC1> _Dst(reinterpret_cast<Pixel16scC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32sc16sc_C2(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                                     size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax, MppiSize aSizeROI,
                                                     MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC2> _Src1(reinterpret_cast<Pixel32scC2 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC2> _Dst(reinterpret_cast<Pixel16scC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32sc16sc_C3(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                                     size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax, MppiSize aSizeROI,
                                                     MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC3> _Src1(reinterpret_cast<Pixel32scC3 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC3> _Dst(reinterpret_cast<Pixel16scC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32sc16sc_C4(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                                     size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax, MppiSize aSizeROI,
                                                     MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC4> _Src1(reinterpret_cast<Pixel32scC4 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC4> _Dst(reinterpret_cast<Pixel16scC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32sc16sc_C1(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step,
                                                          DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                                          MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC1> _Src1(reinterpret_cast<Pixel32scC1 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC1> _Dst(reinterpret_cast<Pixel16scC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32sc16sc_C2(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step,
                                                          DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                                          MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC2> _Src1(reinterpret_cast<Pixel32scC2 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC2> _Dst(reinterpret_cast<Pixel16scC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32sc16sc_C3(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step,
                                                          DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                                          MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC3> _Src1(reinterpret_cast<Pixel32scC3 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC3> _Dst(reinterpret_cast<Pixel16scC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVRToVFR_32sc16sc_C4(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step,
                                                          DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                                          MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC4> _Src1(reinterpret_cast<Pixel32scC4 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC4> _Dst(reinterpret_cast<Pixel16scC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32sc16sc_C1(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                       Mpp32s aSrcMax, DevPtrMpp16sc aDst, size_t aDstStep,
                                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                       CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC1> _Src1(reinterpret_cast<Pixel32scC1 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC1> _Dst(reinterpret_cast<Pixel16scC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32sc16sc_C2(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                       Mpp32s aSrcMax, DevPtrMpp16sc aDst, size_t aDstStep,
                                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                       CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC2> _Src1(reinterpret_cast<Pixel32scC2 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC2> _Dst(reinterpret_cast<Pixel16scC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32sc16sc_C3(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                       Mpp32s aSrcMax, DevPtrMpp16sc aDst, size_t aDstStep,
                                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                       CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC3> _Src1(reinterpret_cast<Pixel32scC3 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC3> _Dst(reinterpret_cast<Pixel16scC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleToFVR_32sc16sc_C4(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                       Mpp32s aSrcMax, DevPtrMpp16sc aDst, size_t aDstStep,
                                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                                       CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC4> _Src1(reinterpret_cast<Pixel32scC4 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16scC4> _Dst(reinterpret_cast<Pixel16scC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, static_cast<RoundingMode>(aRoundingMode), *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32sc32fc_C1(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                                    size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC1> _Src1(reinterpret_cast<Pixel32scC1 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel32fcC1> _Dst(reinterpret_cast<Pixel32fcC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32sc32fc_C2(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                                    size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC2> _Src1(reinterpret_cast<Pixel32scC2 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel32fcC2> _Dst(reinterpret_cast<Pixel32fcC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32sc32fc_C3(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                                    size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC3> _Src1(reinterpret_cast<Pixel32scC3 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel32fcC3> _Dst(reinterpret_cast<Pixel32fcC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciConvert_32sc32fc_C4(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                                    size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC4> _Src1(reinterpret_cast<Pixel32scC4 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel32fcC4> _Dst(reinterpret_cast<Pixel32fcC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Convert(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciScale_32sc32fc_C1(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                  Mpp32s aSrcMax, DevPtrMpp32fc aDst, size_t aDstStep, Mpp32f aDstMin,
                                                  Mpp32f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC1> _Src1(reinterpret_cast<Pixel32scC1 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel32fcC1> _Dst(reinterpret_cast<Pixel32fcC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32sc32fc_C2(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                  Mpp32s aSrcMax, DevPtrMpp32fc aDst, size_t aDstStep, Mpp32f aDstMin,
                                                  Mpp32f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC2> _Src1(reinterpret_cast<Pixel32scC2 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel32fcC2> _Dst(reinterpret_cast<Pixel32fcC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32sc32fc_C3(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                  Mpp32s aSrcMax, DevPtrMpp32fc aDst, size_t aDstStep, Mpp32f aDstMin,
                                                  Mpp32f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC3> _Src1(reinterpret_cast<Pixel32scC3 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel32fcC3> _Dst(reinterpret_cast<Pixel32fcC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScale_32sc32fc_C4(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin,
                                                  Mpp32s aSrcMax, DevPtrMpp32fc aDst, size_t aDstStep, Mpp32f aDstMin,
                                                  Mpp32f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC4> _Src1(reinterpret_cast<Pixel32scC4 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel32fcC4> _Dst(reinterpret_cast<Pixel32fcC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(aSrcMin, aSrcMax, _Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32sc32fc_C1(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                                     size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax, MppiSize aSizeROI,
                                                     CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC1> _Src1(reinterpret_cast<Pixel32scC1 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel32fcC1> _Dst(reinterpret_cast<Pixel32fcC1 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32sc32fc_C2(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                                     size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax, MppiSize aSizeROI,
                                                     CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC2> _Src1(reinterpret_cast<Pixel32scC2 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel32fcC2> _Dst(reinterpret_cast<Pixel32fcC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32sc32fc_C3(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                                     size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax, MppiSize aSizeROI,
                                                     CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC3> _Src1(reinterpret_cast<Pixel32scC3 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel32fcC3> _Dst(reinterpret_cast<Pixel32fcC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciScaleFVR_32sc32fc_C4(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                                     size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax, MppiSize aSizeROI,
                                                     CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel32scC4> _Src1(reinterpret_cast<Pixel32scC4 *>(const_cast<DevPtrMpp32sc>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel32fcC4> _Dst(reinterpret_cast<Pixel32fcC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Scale(_Dst, aDstMin, aDstMax, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
} // extern "C"
// NOLINTEND(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp,modernize-use-using,cppcoreguidelines-pro-type-const-cast,misc-include-cleaner)
