#include "../hsv_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRGBtoHSVSrc(16f);
ForAllChannelsWithAlphaInvokeRGBtoHSVP3Src(16f);
ForAllChannelsWithAlphaInvokeRGBtoHSVP4Src(16f);
ForAllChannelsWithAlphaInvokeRGBtoHSVP3SrcP3(16f);
ForAllChannelsWithAlphaInvokeRGBtoHSVSrcP3(16f);
ForAllChannelsWithAlphaInvokeRGBtoHSVSrcP4(16f);
ForAllChannelsWithAlphaInvokeRGBtoHSVInplace(16f);

ForAllChannelsWithAlphaInvokeHSVtoRGBSrc(16f);
ForAllChannelsWithAlphaInvokeHSVtoRGBP3Src(16f);
ForAllChannelsWithAlphaInvokeHSVtoRGBP4Src(16f);
ForAllChannelsWithAlphaInvokeHSVtoRGBP3SrcP3(16f);
ForAllChannelsWithAlphaInvokeHSVtoRGBSrcP3(16f);
ForAllChannelsWithAlphaInvokeHSVtoRGBSrcP4(16f);
ForAllChannelsWithAlphaInvokeHSVtoRGBInplace(16f);

ForAllChannelsWithAlphaInvokeBGRtoHSVSrc(16f);
ForAllChannelsWithAlphaInvokeBGRtoHSVP3Src(16f);
ForAllChannelsWithAlphaInvokeBGRtoHSVP4Src(16f);
ForAllChannelsWithAlphaInvokeBGRtoHSVP3SrcP3(16f);
ForAllChannelsWithAlphaInvokeBGRtoHSVSrcP3(16f);
ForAllChannelsWithAlphaInvokeBGRtoHSVSrcP4(16f);
ForAllChannelsWithAlphaInvokeBGRtoHSVInplace(16f);

ForAllChannelsWithAlphaInvokeHSVtoBGRSrc(16f);
ForAllChannelsWithAlphaInvokeHSVtoBGRP3Src(16f);
ForAllChannelsWithAlphaInvokeHSVtoBGRP4Src(16f);
ForAllChannelsWithAlphaInvokeHSVtoBGRP3SrcP3(16f);
ForAllChannelsWithAlphaInvokeHSVtoBGRSrcP3(16f);
ForAllChannelsWithAlphaInvokeHSVtoBGRSrcP4(16f);
ForAllChannelsWithAlphaInvokeHSVtoBGRInplace(16f);
} // namespace mpp::image::cuda
