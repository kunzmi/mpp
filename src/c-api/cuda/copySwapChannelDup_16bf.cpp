// deactivate the DLL export macros in imageView:
#include <backends/cuda/image/dllexport_cudai.h>
#undef MPPEXPORT_CUDAI
#define MPPEXPORT_CUDAI
#define ENABLE_CUDA_BFLOAT
// NOLINTBEGIN(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp,modernize-use-using,cppcoreguidelines-pro-type-const-cast,misc-include-cleaner)
#include "catch_and_return.h"
#include "dllexport.h"
#include "mppc_capi_defs.h"
#include <algorithm>
#include <backends/cuda/image/imageView.h>
#include <backends/cuda/image/imageView_dataExchangeAndInit_impl.h> //NOLINT(misc-include-cleaner)
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
    MPPErrorCode DLLEXPORT mppciCopy_16bf_C1C2C(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                                size_t aDstStep, Mpp32s aDstChannel, MppiSize aSizeROI,
                                                CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel16bfC1> _Src1(reinterpret_cast<Pixel16bfC1 *>(const_cast<DevPtrMpp16bf>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC2> _Dst(reinterpret_cast<Pixel16bfC2 *>(aDst), {_SizeROI, aDstStep});
            const Channel _DstChannel(aDstChannel);
            _Src1.Copy(_Dst, _DstChannel, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciCopy_16bf_C1C3C(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                                size_t aDstStep, Mpp32s aDstChannel, MppiSize aSizeROI,
                                                CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel16bfC1> _Src1(reinterpret_cast<Pixel16bfC1 *>(const_cast<DevPtrMpp16bf>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC3> _Dst(reinterpret_cast<Pixel16bfC3 *>(aDst), {_SizeROI, aDstStep});
            const Channel _DstChannel(aDstChannel);
            _Src1.Copy(_Dst, _DstChannel, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciCopy_16bf_C1C4C(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                                size_t aDstStep, Mpp32s aDstChannel, MppiSize aSizeROI,
                                                CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel16bfC1> _Src1(reinterpret_cast<Pixel16bfC1 *>(const_cast<DevPtrMpp16bf>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC4> _Dst(reinterpret_cast<Pixel16bfC4 *>(aDst), {_SizeROI, aDstStep});
            const Channel _DstChannel(aDstChannel);
            _Src1.Copy(_Dst, _DstChannel, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciCopy_16bf_C2C1C(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel,
                                                DevPtrMpp16bf aDst, size_t aDstStep, MppiSize aSizeROI,
                                                CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel16bfC2> _Src1(reinterpret_cast<Pixel16bfC2 *>(const_cast<DevPtrMpp16bf>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC1> _Dst(reinterpret_cast<Pixel16bfC1 *>(aDst), {_SizeROI, aDstStep});
            const Channel _SrcChannel(aSrcChannel);
            _Src1.Copy(_SrcChannel, _Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciCopy_16bf_C3C1C(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel,
                                                DevPtrMpp16bf aDst, size_t aDstStep, MppiSize aSizeROI,
                                                CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel16bfC3> _Src1(reinterpret_cast<Pixel16bfC3 *>(const_cast<DevPtrMpp16bf>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC1> _Dst(reinterpret_cast<Pixel16bfC1 *>(aDst), {_SizeROI, aDstStep});
            const Channel _SrcChannel(aSrcChannel);
            _Src1.Copy(_SrcChannel, _Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciCopy_16bf_C4C1C(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel,
                                                DevPtrMpp16bf aDst, size_t aDstStep, MppiSize aSizeROI,
                                                CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel16bfC4> _Src1(reinterpret_cast<Pixel16bfC4 *>(const_cast<DevPtrMpp16bf>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC1> _Dst(reinterpret_cast<Pixel16bfC1 *>(aDst), {_SizeROI, aDstStep});
            const Channel _SrcChannel(aSrcChannel);
            _Src1.Copy(_SrcChannel, _Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciCopy_16bf_C2C(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel,
                                              DevPtrMpp16bf aDst, size_t aDstStep, Mpp32s aDstChannel,
                                              MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel16bfC2> _Src1(reinterpret_cast<Pixel16bfC2 *>(const_cast<DevPtrMpp16bf>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC2> _Dst(reinterpret_cast<Pixel16bfC2 *>(aDst), {_SizeROI, aDstStep});
            const Channel _SrcChannel(aSrcChannel);
            const Channel _DstChannel(aDstChannel);
            _Src1.Copy(_SrcChannel, _Dst, _DstChannel, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciCopy_16bf_C2C3C(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel,
                                                DevPtrMpp16bf aDst, size_t aDstStep, Mpp32s aDstChannel,
                                                MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel16bfC2> _Src1(reinterpret_cast<Pixel16bfC2 *>(const_cast<DevPtrMpp16bf>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC3> _Dst(reinterpret_cast<Pixel16bfC3 *>(aDst), {_SizeROI, aDstStep});
            const Channel _SrcChannel(aSrcChannel);
            const Channel _DstChannel(aDstChannel);
            _Src1.Copy(_SrcChannel, _Dst, _DstChannel, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciCopy_16bf_C2C4C(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel,
                                                DevPtrMpp16bf aDst, size_t aDstStep, Mpp32s aDstChannel,
                                                MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel16bfC2> _Src1(reinterpret_cast<Pixel16bfC2 *>(const_cast<DevPtrMpp16bf>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC4> _Dst(reinterpret_cast<Pixel16bfC4 *>(aDst), {_SizeROI, aDstStep});
            const Channel _SrcChannel(aSrcChannel);
            const Channel _DstChannel(aDstChannel);
            _Src1.Copy(_SrcChannel, _Dst, _DstChannel, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciCopy_16bf_C3C2C(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel,
                                                DevPtrMpp16bf aDst, size_t aDstStep, Mpp32s aDstChannel,
                                                MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel16bfC3> _Src1(reinterpret_cast<Pixel16bfC3 *>(const_cast<DevPtrMpp16bf>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC2> _Dst(reinterpret_cast<Pixel16bfC2 *>(aDst), {_SizeROI, aDstStep});
            const Channel _SrcChannel(aSrcChannel);
            const Channel _DstChannel(aDstChannel);
            _Src1.Copy(_SrcChannel, _Dst, _DstChannel, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciCopy_16bf_C3C(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel,
                                              DevPtrMpp16bf aDst, size_t aDstStep, Mpp32s aDstChannel,
                                              MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel16bfC3> _Src1(reinterpret_cast<Pixel16bfC3 *>(const_cast<DevPtrMpp16bf>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC3> _Dst(reinterpret_cast<Pixel16bfC3 *>(aDst), {_SizeROI, aDstStep});
            const Channel _SrcChannel(aSrcChannel);
            const Channel _DstChannel(aDstChannel);
            _Src1.Copy(_SrcChannel, _Dst, _DstChannel, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciCopy_16bf_C3C4C(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel,
                                                DevPtrMpp16bf aDst, size_t aDstStep, Mpp32s aDstChannel,
                                                MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel16bfC3> _Src1(reinterpret_cast<Pixel16bfC3 *>(const_cast<DevPtrMpp16bf>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC4> _Dst(reinterpret_cast<Pixel16bfC4 *>(aDst), {_SizeROI, aDstStep});
            const Channel _SrcChannel(aSrcChannel);
            const Channel _DstChannel(aDstChannel);
            _Src1.Copy(_SrcChannel, _Dst, _DstChannel, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciCopy_16bf_C4C2C(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel,
                                                DevPtrMpp16bf aDst, size_t aDstStep, Mpp32s aDstChannel,
                                                MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel16bfC4> _Src1(reinterpret_cast<Pixel16bfC4 *>(const_cast<DevPtrMpp16bf>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC2> _Dst(reinterpret_cast<Pixel16bfC2 *>(aDst), {_SizeROI, aDstStep});
            const Channel _SrcChannel(aSrcChannel);
            const Channel _DstChannel(aDstChannel);
            _Src1.Copy(_SrcChannel, _Dst, _DstChannel, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciCopy_16bf_C4C3C(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel,
                                                DevPtrMpp16bf aDst, size_t aDstStep, Mpp32s aDstChannel,
                                                MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel16bfC4> _Src1(reinterpret_cast<Pixel16bfC4 *>(const_cast<DevPtrMpp16bf>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC3> _Dst(reinterpret_cast<Pixel16bfC3 *>(aDst), {_SizeROI, aDstStep});
            const Channel _SrcChannel(aSrcChannel);
            const Channel _DstChannel(aDstChannel);
            _Src1.Copy(_SrcChannel, _Dst, _DstChannel, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }
    MPPErrorCode DLLEXPORT mppciCopy_16bf_C4C(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel,
                                              DevPtrMpp16bf aDst, size_t aDstStep, Mpp32s aDstChannel,
                                              MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel16bfC4> _Src1(reinterpret_cast<Pixel16bfC4 *>(const_cast<DevPtrMpp16bf>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC4> _Dst(reinterpret_cast<Pixel16bfC4 *>(aDst), {_SizeROI, aDstStep});
            const Channel _SrcChannel(aSrcChannel);
            const Channel _DstChannel(aDstChannel);
            _Src1.Copy(_SrcChannel, _Dst, _DstChannel, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciSwapChannel_16bf_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                                    size_t aDstStep, const Mpp32s aDstChannels[3], MppiSize aSizeROI,
                                                    CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel16bfC3> _Src1(reinterpret_cast<Pixel16bfC3 *>(const_cast<DevPtrMpp16bf>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC3> _Dst(reinterpret_cast<Pixel16bfC3 *>(aDst), {_SizeROI, aDstStep});
            const ChannelList<3> _DstChannels(aDstChannels);
            _Src1.SwapChannel(_Dst, _DstChannels, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciSwapChannel_16bf_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                                    size_t aDstStep, const Mpp32s aDstChannels[4], MppiSize aSizeROI,
                                                    CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel16bfC4> _Src1(reinterpret_cast<Pixel16bfC4 *>(const_cast<DevPtrMpp16bf>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC4> _Dst(reinterpret_cast<Pixel16bfC4 *>(aDst), {_SizeROI, aDstStep});
            const ChannelList<4> _DstChannels(aDstChannels);
            _Src1.SwapChannel(_Dst, _DstChannels, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciSwapChannel_16bf_C4C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                                      size_t aDstStep, const Mpp32s aDstChannels[3], MppiSize aSizeROI,
                                                      CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel16bfC4> _Src1(reinterpret_cast<Pixel16bfC4 *>(const_cast<DevPtrMpp16bf>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC3> _Dst(reinterpret_cast<Pixel16bfC3 *>(aDst), {_SizeROI, aDstStep});
            const ChannelList<3> _DstChannels(aDstChannels);
            _Src1.SwapChannel(_Dst, _DstChannels, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciSwapChannel_16bf_C3C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                                      size_t aDstStep, const Mpp32s aDstChannels[3], Mpp16bf aValue,
                                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel16bfC3> _Src1(reinterpret_cast<Pixel16bfC3 *>(const_cast<DevPtrMpp16bf>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC4> _Dst(reinterpret_cast<Pixel16bfC4 *>(aDst), {_SizeROI, aDstStep});
            const ChannelList<4> _DstChannels(aDstChannels);
            _Src1.SwapChannel(_Dst, _DstChannels, aValue, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciDup_16bf_C1C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                              size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel16bfC1> _Src1(reinterpret_cast<Pixel16bfC1 *>(const_cast<DevPtrMpp16bf>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC2> _Dst(reinterpret_cast<Pixel16bfC2 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Dup(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciDup_16bf_C1C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                              size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel16bfC1> _Src1(reinterpret_cast<Pixel16bfC1 *>(const_cast<DevPtrMpp16bf>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC3> _Dst(reinterpret_cast<Pixel16bfC3 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Dup(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciDup_16bf_C1C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                              size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel16bfC1> _Src1(reinterpret_cast<Pixel16bfC1 *>(const_cast<DevPtrMpp16bf>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC4> _Dst(reinterpret_cast<Pixel16bfC4 *>(aDst), {_SizeROI, aDstStep});
            _Src1.Dup(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

    MPPErrorCode DLLEXPORT mppciDup_16bf_C1AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                               size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx)
    {
        try
        {
            const StreamCtx *_StreamCtx =
                aStreamCtx == nullptr ? &StreamCtxSingleton::Get() : reinterpret_cast<const StreamCtx *>(aStreamCtx);
            const Size2D _SizeROI(aSizeROI.width, aSizeROI.height);
            const ImageView<Pixel16bfC1> _Src1(reinterpret_cast<Pixel16bfC1 *>(const_cast<DevPtrMpp16bf>(aSrc1)),
                                               {_SizeROI, aSrc1Step});
            ImageView<Pixel16bfC4A> _Dst(reinterpret_cast<Pixel16bfC4A *>(aDst), {_SizeROI, aDstStep});
            _Src1.Dup(_Dst, *_StreamCtx);
        }
        CATCH_AND_RETURN_ERRORCODE;
    }

} // extern "C"
// NOLINTEND(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp,modernize-use-using,cppcoreguidelines-pro-type-const-cast,misc-include-cleaner)
