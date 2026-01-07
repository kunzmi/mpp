#include "../luv_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRGBtoLUVSrc(16u);
ForAllChannelsWithAlphaInvokeRGBtoLUVP3Src(16u);
ForAllChannelsWithAlphaInvokeRGBtoLUVP4Src(16u);
ForAllChannelsWithAlphaInvokeRGBtoLUVP3SrcP3(16u);
ForAllChannelsWithAlphaInvokeRGBtoLUVSrcP3(16u);
ForAllChannelsWithAlphaInvokeRGBtoLUVSrcP4(16u);
ForAllChannelsWithAlphaInvokeRGBtoLUVInplace(16u);

ForAllChannelsWithAlphaInvokeLUVtoRGBSrc(16u);
ForAllChannelsWithAlphaInvokeLUVtoRGBP3Src(16u);
ForAllChannelsWithAlphaInvokeLUVtoRGBP4Src(16u);
ForAllChannelsWithAlphaInvokeLUVtoRGBP3SrcP3(16u);
ForAllChannelsWithAlphaInvokeLUVtoRGBSrcP3(16u);
ForAllChannelsWithAlphaInvokeLUVtoRGBSrcP4(16u);
ForAllChannelsWithAlphaInvokeLUVtoRGBInplace(16u);

ForAllChannelsWithAlphaInvokeBGRtoLUVSrc(16u);
ForAllChannelsWithAlphaInvokeBGRtoLUVP3Src(16u);
ForAllChannelsWithAlphaInvokeBGRtoLUVP4Src(16u);
ForAllChannelsWithAlphaInvokeBGRtoLUVP3SrcP3(16u);
ForAllChannelsWithAlphaInvokeBGRtoLUVSrcP3(16u);
ForAllChannelsWithAlphaInvokeBGRtoLUVSrcP4(16u);
ForAllChannelsWithAlphaInvokeBGRtoLUVInplace(16u);

ForAllChannelsWithAlphaInvokeLUVtoBGRSrc(16u);
ForAllChannelsWithAlphaInvokeLUVtoBGRP3Src(16u);
ForAllChannelsWithAlphaInvokeLUVtoBGRP4Src(16u);
ForAllChannelsWithAlphaInvokeLUVtoBGRP3SrcP3(16u);
ForAllChannelsWithAlphaInvokeLUVtoBGRSrcP3(16u);
ForAllChannelsWithAlphaInvokeLUVtoBGRSrcP4(16u);
ForAllChannelsWithAlphaInvokeLUVtoBGRInplace(16u);
} // namespace mpp::image::cuda
