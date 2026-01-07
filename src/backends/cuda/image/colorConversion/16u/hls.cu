#include "../hls_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRGBtoHLSSrc(16u);
ForAllChannelsWithAlphaInvokeRGBtoHLSP3Src(16u);
ForAllChannelsWithAlphaInvokeRGBtoHLSP4Src(16u);
ForAllChannelsWithAlphaInvokeRGBtoHLSP3SrcP3(16u);
ForAllChannelsWithAlphaInvokeRGBtoHLSSrcP3(16u);
ForAllChannelsWithAlphaInvokeRGBtoHLSSrcP4(16u);
ForAllChannelsWithAlphaInvokeRGBtoHLSInplace(16u);

ForAllChannelsWithAlphaInvokeHLStoRGBSrc(16u);
ForAllChannelsWithAlphaInvokeHLStoRGBP3Src(16u);
ForAllChannelsWithAlphaInvokeHLStoRGBP4Src(16u);
ForAllChannelsWithAlphaInvokeHLStoRGBP3SrcP3(16u);
ForAllChannelsWithAlphaInvokeHLStoRGBSrcP3(16u);
ForAllChannelsWithAlphaInvokeHLStoRGBSrcP4(16u);
ForAllChannelsWithAlphaInvokeHLStoRGBInplace(16u);

ForAllChannelsWithAlphaInvokeBGRtoHLSSrc(16u);
ForAllChannelsWithAlphaInvokeBGRtoHLSP3Src(16u);
ForAllChannelsWithAlphaInvokeBGRtoHLSP4Src(16u);
ForAllChannelsWithAlphaInvokeBGRtoHLSP3SrcP3(16u);
ForAllChannelsWithAlphaInvokeBGRtoHLSSrcP3(16u);
ForAllChannelsWithAlphaInvokeBGRtoHLSSrcP4(16u);
ForAllChannelsWithAlphaInvokeBGRtoHLSInplace(16u);

ForAllChannelsWithAlphaInvokeHLStoBGRSrc(16u);
ForAllChannelsWithAlphaInvokeHLStoBGRP3Src(16u);
ForAllChannelsWithAlphaInvokeHLStoBGRP4Src(16u);
ForAllChannelsWithAlphaInvokeHLStoBGRP3SrcP3(16u);
ForAllChannelsWithAlphaInvokeHLStoBGRSrcP3(16u);
ForAllChannelsWithAlphaInvokeHLStoBGRSrcP4(16u);
ForAllChannelsWithAlphaInvokeHLStoBGRInplace(16u);
} // namespace mpp::image::cuda
