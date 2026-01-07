#include "../luv_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRGBtoLUVSrc(32f);
ForAllChannelsWithAlphaInvokeRGBtoLUVP3Src(32f);
ForAllChannelsWithAlphaInvokeRGBtoLUVP4Src(32f);
ForAllChannelsWithAlphaInvokeRGBtoLUVP3SrcP3(32f);
ForAllChannelsWithAlphaInvokeRGBtoLUVSrcP3(32f);
ForAllChannelsWithAlphaInvokeRGBtoLUVSrcP4(32f);
ForAllChannelsWithAlphaInvokeRGBtoLUVInplace(32f);

ForAllChannelsWithAlphaInvokeLUVtoRGBSrc(32f);
ForAllChannelsWithAlphaInvokeLUVtoRGBP3Src(32f);
ForAllChannelsWithAlphaInvokeLUVtoRGBP4Src(32f);
ForAllChannelsWithAlphaInvokeLUVtoRGBP3SrcP3(32f);
ForAllChannelsWithAlphaInvokeLUVtoRGBSrcP3(32f);
ForAllChannelsWithAlphaInvokeLUVtoRGBSrcP4(32f);
ForAllChannelsWithAlphaInvokeLUVtoRGBInplace(32f);

ForAllChannelsWithAlphaInvokeBGRtoLUVSrc(32f);
ForAllChannelsWithAlphaInvokeBGRtoLUVP3Src(32f);
ForAllChannelsWithAlphaInvokeBGRtoLUVP4Src(32f);
ForAllChannelsWithAlphaInvokeBGRtoLUVP3SrcP3(32f);
ForAllChannelsWithAlphaInvokeBGRtoLUVSrcP3(32f);
ForAllChannelsWithAlphaInvokeBGRtoLUVSrcP4(32f);
ForAllChannelsWithAlphaInvokeBGRtoLUVInplace(32f);

ForAllChannelsWithAlphaInvokeLUVtoBGRSrc(32f);
ForAllChannelsWithAlphaInvokeLUVtoBGRP3Src(32f);
ForAllChannelsWithAlphaInvokeLUVtoBGRP4Src(32f);
ForAllChannelsWithAlphaInvokeLUVtoBGRP3SrcP3(32f);
ForAllChannelsWithAlphaInvokeLUVtoBGRSrcP3(32f);
ForAllChannelsWithAlphaInvokeLUVtoBGRSrcP4(32f);
ForAllChannelsWithAlphaInvokeLUVtoBGRInplace(32f);
} // namespace mpp::image::cuda
