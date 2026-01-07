#include "../hsv_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRGBtoHSVSrc(8u);
ForAllChannelsWithAlphaInvokeRGBtoHSVP3Src(8u);
ForAllChannelsWithAlphaInvokeRGBtoHSVP4Src(8u);
ForAllChannelsWithAlphaInvokeRGBtoHSVP3SrcP3(8u);
ForAllChannelsWithAlphaInvokeRGBtoHSVSrcP3(8u);
ForAllChannelsWithAlphaInvokeRGBtoHSVSrcP4(8u);
ForAllChannelsWithAlphaInvokeRGBtoHSVInplace(8u);

ForAllChannelsWithAlphaInvokeHSVtoRGBSrc(8u);
ForAllChannelsWithAlphaInvokeHSVtoRGBP3Src(8u);
ForAllChannelsWithAlphaInvokeHSVtoRGBP4Src(8u);
ForAllChannelsWithAlphaInvokeHSVtoRGBP3SrcP3(8u);
ForAllChannelsWithAlphaInvokeHSVtoRGBSrcP3(8u);
ForAllChannelsWithAlphaInvokeHSVtoRGBSrcP4(8u);
ForAllChannelsWithAlphaInvokeHSVtoRGBInplace(8u);

ForAllChannelsWithAlphaInvokeBGRtoHSVSrc(8u);
ForAllChannelsWithAlphaInvokeBGRtoHSVP3Src(8u);
ForAllChannelsWithAlphaInvokeBGRtoHSVP4Src(8u);
ForAllChannelsWithAlphaInvokeBGRtoHSVP3SrcP3(8u);
ForAllChannelsWithAlphaInvokeBGRtoHSVSrcP3(8u);
ForAllChannelsWithAlphaInvokeBGRtoHSVSrcP4(8u);
ForAllChannelsWithAlphaInvokeBGRtoHSVInplace(8u);

ForAllChannelsWithAlphaInvokeHSVtoBGRSrc(8u);
ForAllChannelsWithAlphaInvokeHSVtoBGRP3Src(8u);
ForAllChannelsWithAlphaInvokeHSVtoBGRP4Src(8u);
ForAllChannelsWithAlphaInvokeHSVtoBGRP3SrcP3(8u);
ForAllChannelsWithAlphaInvokeHSVtoBGRSrcP3(8u);
ForAllChannelsWithAlphaInvokeHSVtoBGRSrcP4(8u);
ForAllChannelsWithAlphaInvokeHSVtoBGRInplace(8u);
} // namespace mpp::image::cuda
