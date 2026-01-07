#include "../hsv_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRGBtoHSVSrc(32f);
ForAllChannelsWithAlphaInvokeRGBtoHSVP3Src(32f);
ForAllChannelsWithAlphaInvokeRGBtoHSVP4Src(32f);
ForAllChannelsWithAlphaInvokeRGBtoHSVP3SrcP3(32f);
ForAllChannelsWithAlphaInvokeRGBtoHSVSrcP3(32f);
ForAllChannelsWithAlphaInvokeRGBtoHSVSrcP4(32f);
ForAllChannelsWithAlphaInvokeRGBtoHSVInplace(32f);

ForAllChannelsWithAlphaInvokeHSVtoRGBSrc(32f);
ForAllChannelsWithAlphaInvokeHSVtoRGBP3Src(32f);
ForAllChannelsWithAlphaInvokeHSVtoRGBP4Src(32f);
ForAllChannelsWithAlphaInvokeHSVtoRGBP3SrcP3(32f);
ForAllChannelsWithAlphaInvokeHSVtoRGBSrcP3(32f);
ForAllChannelsWithAlphaInvokeHSVtoRGBSrcP4(32f);
ForAllChannelsWithAlphaInvokeHSVtoRGBInplace(32f);

ForAllChannelsWithAlphaInvokeBGRtoHSVSrc(32f);
ForAllChannelsWithAlphaInvokeBGRtoHSVP3Src(32f);
ForAllChannelsWithAlphaInvokeBGRtoHSVP4Src(32f);
ForAllChannelsWithAlphaInvokeBGRtoHSVP3SrcP3(32f);
ForAllChannelsWithAlphaInvokeBGRtoHSVSrcP3(32f);
ForAllChannelsWithAlphaInvokeBGRtoHSVSrcP4(32f);
ForAllChannelsWithAlphaInvokeBGRtoHSVInplace(32f);

ForAllChannelsWithAlphaInvokeHSVtoBGRSrc(32f);
ForAllChannelsWithAlphaInvokeHSVtoBGRP3Src(32f);
ForAllChannelsWithAlphaInvokeHSVtoBGRP4Src(32f);
ForAllChannelsWithAlphaInvokeHSVtoBGRP3SrcP3(32f);
ForAllChannelsWithAlphaInvokeHSVtoBGRSrcP3(32f);
ForAllChannelsWithAlphaInvokeHSVtoBGRSrcP4(32f);
ForAllChannelsWithAlphaInvokeHSVtoBGRInplace(32f);
} // namespace mpp::image::cuda
