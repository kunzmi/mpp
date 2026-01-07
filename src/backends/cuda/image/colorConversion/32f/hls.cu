#include "../hls_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRGBtoHLSSrc(32f);
ForAllChannelsWithAlphaInvokeRGBtoHLSP3Src(32f);
ForAllChannelsWithAlphaInvokeRGBtoHLSP4Src(32f);
ForAllChannelsWithAlphaInvokeRGBtoHLSP3SrcP3(32f);
ForAllChannelsWithAlphaInvokeRGBtoHLSSrcP3(32f);
ForAllChannelsWithAlphaInvokeRGBtoHLSSrcP4(32f);
ForAllChannelsWithAlphaInvokeRGBtoHLSInplace(32f);

ForAllChannelsWithAlphaInvokeHLStoRGBSrc(32f);
ForAllChannelsWithAlphaInvokeHLStoRGBP3Src(32f);
ForAllChannelsWithAlphaInvokeHLStoRGBP4Src(32f);
ForAllChannelsWithAlphaInvokeHLStoRGBP3SrcP3(32f);
ForAllChannelsWithAlphaInvokeHLStoRGBSrcP3(32f);
ForAllChannelsWithAlphaInvokeHLStoRGBSrcP4(32f);
ForAllChannelsWithAlphaInvokeHLStoRGBInplace(32f);

ForAllChannelsWithAlphaInvokeBGRtoHLSSrc(32f);
ForAllChannelsWithAlphaInvokeBGRtoHLSP3Src(32f);
ForAllChannelsWithAlphaInvokeBGRtoHLSP4Src(32f);
ForAllChannelsWithAlphaInvokeBGRtoHLSP3SrcP3(32f);
ForAllChannelsWithAlphaInvokeBGRtoHLSSrcP3(32f);
ForAllChannelsWithAlphaInvokeBGRtoHLSSrcP4(32f);
ForAllChannelsWithAlphaInvokeBGRtoHLSInplace(32f);

ForAllChannelsWithAlphaInvokeHLStoBGRSrc(32f);
ForAllChannelsWithAlphaInvokeHLStoBGRP3Src(32f);
ForAllChannelsWithAlphaInvokeHLStoBGRP4Src(32f);
ForAllChannelsWithAlphaInvokeHLStoBGRP3SrcP3(32f);
ForAllChannelsWithAlphaInvokeHLStoBGRSrcP3(32f);
ForAllChannelsWithAlphaInvokeHLStoBGRSrcP4(32f);
ForAllChannelsWithAlphaInvokeHLStoBGRInplace(32f);
} // namespace mpp::image::cuda
