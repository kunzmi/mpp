#include "../luv_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRGBtoLUVSrc(8u);
ForAllChannelsWithAlphaInvokeRGBtoLUVP3Src(8u);
ForAllChannelsWithAlphaInvokeRGBtoLUVP4Src(8u);
ForAllChannelsWithAlphaInvokeRGBtoLUVP3SrcP3(8u);
ForAllChannelsWithAlphaInvokeRGBtoLUVSrcP3(8u);
ForAllChannelsWithAlphaInvokeRGBtoLUVSrcP4(8u);
ForAllChannelsWithAlphaInvokeRGBtoLUVInplace(8u);

ForAllChannelsWithAlphaInvokeLUVtoRGBSrc(8u);
ForAllChannelsWithAlphaInvokeLUVtoRGBP3Src(8u);
ForAllChannelsWithAlphaInvokeLUVtoRGBP4Src(8u);
ForAllChannelsWithAlphaInvokeLUVtoRGBP3SrcP3(8u);
ForAllChannelsWithAlphaInvokeLUVtoRGBSrcP3(8u);
ForAllChannelsWithAlphaInvokeLUVtoRGBSrcP4(8u);
ForAllChannelsWithAlphaInvokeLUVtoRGBInplace(8u);

ForAllChannelsWithAlphaInvokeBGRtoLUVSrc(8u);
ForAllChannelsWithAlphaInvokeBGRtoLUVP3Src(8u);
ForAllChannelsWithAlphaInvokeBGRtoLUVP4Src(8u);
ForAllChannelsWithAlphaInvokeBGRtoLUVP3SrcP3(8u);
ForAllChannelsWithAlphaInvokeBGRtoLUVSrcP3(8u);
ForAllChannelsWithAlphaInvokeBGRtoLUVSrcP4(8u);
ForAllChannelsWithAlphaInvokeBGRtoLUVInplace(8u);

ForAllChannelsWithAlphaInvokeLUVtoBGRSrc(8u);
ForAllChannelsWithAlphaInvokeLUVtoBGRP3Src(8u);
ForAllChannelsWithAlphaInvokeLUVtoBGRP4Src(8u);
ForAllChannelsWithAlphaInvokeLUVtoBGRP3SrcP3(8u);
ForAllChannelsWithAlphaInvokeLUVtoBGRSrcP3(8u);
ForAllChannelsWithAlphaInvokeLUVtoBGRSrcP4(8u);
ForAllChannelsWithAlphaInvokeLUVtoBGRInplace(8u);
} // namespace mpp::image::cuda
