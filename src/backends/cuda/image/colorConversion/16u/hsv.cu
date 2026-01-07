#include "../hsv_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRGBtoHSVSrc(16u);
ForAllChannelsWithAlphaInvokeRGBtoHSVP3Src(16u);
ForAllChannelsWithAlphaInvokeRGBtoHSVP4Src(16u);
ForAllChannelsWithAlphaInvokeRGBtoHSVP3SrcP3(16u);
ForAllChannelsWithAlphaInvokeRGBtoHSVSrcP3(16u);
ForAllChannelsWithAlphaInvokeRGBtoHSVSrcP4(16u);
ForAllChannelsWithAlphaInvokeRGBtoHSVInplace(16u);

ForAllChannelsWithAlphaInvokeHSVtoRGBSrc(16u);
ForAllChannelsWithAlphaInvokeHSVtoRGBP3Src(16u);
ForAllChannelsWithAlphaInvokeHSVtoRGBP4Src(16u);
ForAllChannelsWithAlphaInvokeHSVtoRGBP3SrcP3(16u);
ForAllChannelsWithAlphaInvokeHSVtoRGBSrcP3(16u);
ForAllChannelsWithAlphaInvokeHSVtoRGBSrcP4(16u);
ForAllChannelsWithAlphaInvokeHSVtoRGBInplace(16u);

ForAllChannelsWithAlphaInvokeBGRtoHSVSrc(16u);
ForAllChannelsWithAlphaInvokeBGRtoHSVP3Src(16u);
ForAllChannelsWithAlphaInvokeBGRtoHSVP4Src(16u);
ForAllChannelsWithAlphaInvokeBGRtoHSVP3SrcP3(16u);
ForAllChannelsWithAlphaInvokeBGRtoHSVSrcP3(16u);
ForAllChannelsWithAlphaInvokeBGRtoHSVSrcP4(16u);
ForAllChannelsWithAlphaInvokeBGRtoHSVInplace(16u);

ForAllChannelsWithAlphaInvokeHSVtoBGRSrc(16u);
ForAllChannelsWithAlphaInvokeHSVtoBGRP3Src(16u);
ForAllChannelsWithAlphaInvokeHSVtoBGRP4Src(16u);
ForAllChannelsWithAlphaInvokeHSVtoBGRP3SrcP3(16u);
ForAllChannelsWithAlphaInvokeHSVtoBGRSrcP3(16u);
ForAllChannelsWithAlphaInvokeHSVtoBGRSrcP4(16u);
ForAllChannelsWithAlphaInvokeHSVtoBGRInplace(16u);
} // namespace mpp::image::cuda
