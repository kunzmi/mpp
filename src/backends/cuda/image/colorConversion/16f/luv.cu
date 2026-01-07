#include "../luv_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRGBtoLUVSrc(16f);
ForAllChannelsWithAlphaInvokeRGBtoLUVP3Src(16f);
ForAllChannelsWithAlphaInvokeRGBtoLUVP4Src(16f);
ForAllChannelsWithAlphaInvokeRGBtoLUVP3SrcP3(16f);
ForAllChannelsWithAlphaInvokeRGBtoLUVSrcP3(16f);
ForAllChannelsWithAlphaInvokeRGBtoLUVSrcP4(16f);
ForAllChannelsWithAlphaInvokeRGBtoLUVInplace(16f);

ForAllChannelsWithAlphaInvokeLUVtoRGBSrc(16f);
ForAllChannelsWithAlphaInvokeLUVtoRGBP3Src(16f);
ForAllChannelsWithAlphaInvokeLUVtoRGBP4Src(16f);
ForAllChannelsWithAlphaInvokeLUVtoRGBP3SrcP3(16f);
ForAllChannelsWithAlphaInvokeLUVtoRGBSrcP3(16f);
ForAllChannelsWithAlphaInvokeLUVtoRGBSrcP4(16f);
ForAllChannelsWithAlphaInvokeLUVtoRGBInplace(16f);

ForAllChannelsWithAlphaInvokeBGRtoLUVSrc(16f);
ForAllChannelsWithAlphaInvokeBGRtoLUVP3Src(16f);
ForAllChannelsWithAlphaInvokeBGRtoLUVP4Src(16f);
ForAllChannelsWithAlphaInvokeBGRtoLUVP3SrcP3(16f);
ForAllChannelsWithAlphaInvokeBGRtoLUVSrcP3(16f);
ForAllChannelsWithAlphaInvokeBGRtoLUVSrcP4(16f);
ForAllChannelsWithAlphaInvokeBGRtoLUVInplace(16f);

ForAllChannelsWithAlphaInvokeLUVtoBGRSrc(16f);
ForAllChannelsWithAlphaInvokeLUVtoBGRP3Src(16f);
ForAllChannelsWithAlphaInvokeLUVtoBGRP4Src(16f);
ForAllChannelsWithAlphaInvokeLUVtoBGRP3SrcP3(16f);
ForAllChannelsWithAlphaInvokeLUVtoBGRSrcP3(16f);
ForAllChannelsWithAlphaInvokeLUVtoBGRSrcP4(16f);
ForAllChannelsWithAlphaInvokeLUVtoBGRInplace(16f);
} // namespace mpp::image::cuda
