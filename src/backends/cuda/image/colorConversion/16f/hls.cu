#include "../hls_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRGBtoHLSSrc(16f);
ForAllChannelsWithAlphaInvokeRGBtoHLSP3Src(16f);
ForAllChannelsWithAlphaInvokeRGBtoHLSP4Src(16f);
ForAllChannelsWithAlphaInvokeRGBtoHLSP3SrcP3(16f);
ForAllChannelsWithAlphaInvokeRGBtoHLSSrcP3(16f);
ForAllChannelsWithAlphaInvokeRGBtoHLSSrcP4(16f);
ForAllChannelsWithAlphaInvokeRGBtoHLSInplace(16f);

ForAllChannelsWithAlphaInvokeHLStoRGBSrc(16f);
ForAllChannelsWithAlphaInvokeHLStoRGBP3Src(16f);
ForAllChannelsWithAlphaInvokeHLStoRGBP4Src(16f);
ForAllChannelsWithAlphaInvokeHLStoRGBP3SrcP3(16f);
ForAllChannelsWithAlphaInvokeHLStoRGBSrcP3(16f);
ForAllChannelsWithAlphaInvokeHLStoRGBSrcP4(16f);
ForAllChannelsWithAlphaInvokeHLStoRGBInplace(16f);

ForAllChannelsWithAlphaInvokeBGRtoHLSSrc(16f);
ForAllChannelsWithAlphaInvokeBGRtoHLSP3Src(16f);
ForAllChannelsWithAlphaInvokeBGRtoHLSP4Src(16f);
ForAllChannelsWithAlphaInvokeBGRtoHLSP3SrcP3(16f);
ForAllChannelsWithAlphaInvokeBGRtoHLSSrcP3(16f);
ForAllChannelsWithAlphaInvokeBGRtoHLSSrcP4(16f);
ForAllChannelsWithAlphaInvokeBGRtoHLSInplace(16f);

ForAllChannelsWithAlphaInvokeHLStoBGRSrc(16f);
ForAllChannelsWithAlphaInvokeHLStoBGRP3Src(16f);
ForAllChannelsWithAlphaInvokeHLStoBGRP4Src(16f);
ForAllChannelsWithAlphaInvokeHLStoBGRP3SrcP3(16f);
ForAllChannelsWithAlphaInvokeHLStoBGRSrcP3(16f);
ForAllChannelsWithAlphaInvokeHLStoBGRSrcP4(16f);
ForAllChannelsWithAlphaInvokeHLStoBGRInplace(16f);
} // namespace mpp::image::cuda
