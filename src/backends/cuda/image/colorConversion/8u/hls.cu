#include "../hls_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRGBtoHLSSrc(8u);
ForAllChannelsWithAlphaInvokeRGBtoHLSP3Src(8u);
ForAllChannelsWithAlphaInvokeRGBtoHLSP4Src(8u);
ForAllChannelsWithAlphaInvokeRGBtoHLSP3SrcP3(8u);
ForAllChannelsWithAlphaInvokeRGBtoHLSSrcP3(8u);
ForAllChannelsWithAlphaInvokeRGBtoHLSSrcP4(8u);
ForAllChannelsWithAlphaInvokeRGBtoHLSInplace(8u);

ForAllChannelsWithAlphaInvokeHLStoRGBSrc(8u);
ForAllChannelsWithAlphaInvokeHLStoRGBP3Src(8u);
ForAllChannelsWithAlphaInvokeHLStoRGBP4Src(8u);
ForAllChannelsWithAlphaInvokeHLStoRGBP3SrcP3(8u);
ForAllChannelsWithAlphaInvokeHLStoRGBSrcP3(8u);
ForAllChannelsWithAlphaInvokeHLStoRGBSrcP4(8u);
ForAllChannelsWithAlphaInvokeHLStoRGBInplace(8u);

ForAllChannelsWithAlphaInvokeBGRtoHLSSrc(8u);
ForAllChannelsWithAlphaInvokeBGRtoHLSP3Src(8u);
ForAllChannelsWithAlphaInvokeBGRtoHLSP4Src(8u);
ForAllChannelsWithAlphaInvokeBGRtoHLSP3SrcP3(8u);
ForAllChannelsWithAlphaInvokeBGRtoHLSSrcP3(8u);
ForAllChannelsWithAlphaInvokeBGRtoHLSSrcP4(8u);
ForAllChannelsWithAlphaInvokeBGRtoHLSInplace(8u);

ForAllChannelsWithAlphaInvokeHLStoBGRSrc(8u);
ForAllChannelsWithAlphaInvokeHLStoBGRP3Src(8u);
ForAllChannelsWithAlphaInvokeHLStoBGRP4Src(8u);
ForAllChannelsWithAlphaInvokeHLStoBGRP3SrcP3(8u);
ForAllChannelsWithAlphaInvokeHLStoBGRSrcP3(8u);
ForAllChannelsWithAlphaInvokeHLStoBGRSrcP4(8u);
ForAllChannelsWithAlphaInvokeHLStoBGRInplace(8u);
} // namespace mpp::image::cuda
