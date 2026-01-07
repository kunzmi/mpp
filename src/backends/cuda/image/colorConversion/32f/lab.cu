#include "../lab_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRGBtoLabSrc(32f);
ForAllChannelsWithAlphaInvokeRGBtoLabP3Src(32f);
ForAllChannelsWithAlphaInvokeRGBtoLabP4Src(32f);
ForAllChannelsWithAlphaInvokeRGBtoLabP3SrcP3(32f);
ForAllChannelsWithAlphaInvokeRGBtoLabSrcP3(32f);
ForAllChannelsWithAlphaInvokeRGBtoLabSrcP4(32f);
ForAllChannelsWithAlphaInvokeRGBtoLabInplace(32f);

ForAllChannelsWithAlphaInvokeLabtoRGBSrc(32f);
ForAllChannelsWithAlphaInvokeLabtoRGBP3Src(32f);
ForAllChannelsWithAlphaInvokeLabtoRGBP4Src(32f);
ForAllChannelsWithAlphaInvokeLabtoRGBP3SrcP3(32f);
ForAllChannelsWithAlphaInvokeLabtoRGBSrcP3(32f);
ForAllChannelsWithAlphaInvokeLabtoRGBSrcP4(32f);
ForAllChannelsWithAlphaInvokeLabtoRGBInplace(32f);

ForAllChannelsWithAlphaInvokeBGRtoLabSrc(32f);
ForAllChannelsWithAlphaInvokeBGRtoLabP3Src(32f);
ForAllChannelsWithAlphaInvokeBGRtoLabP4Src(32f);
ForAllChannelsWithAlphaInvokeBGRtoLabP3SrcP3(32f);
ForAllChannelsWithAlphaInvokeBGRtoLabSrcP3(32f);
ForAllChannelsWithAlphaInvokeBGRtoLabSrcP4(32f);
ForAllChannelsWithAlphaInvokeBGRtoLabInplace(32f);

ForAllChannelsWithAlphaInvokeLabtoBGRSrc(32f);
ForAllChannelsWithAlphaInvokeLabtoBGRP3Src(32f);
ForAllChannelsWithAlphaInvokeLabtoBGRP4Src(32f);
ForAllChannelsWithAlphaInvokeLabtoBGRP3SrcP3(32f);
ForAllChannelsWithAlphaInvokeLabtoBGRSrcP3(32f);
ForAllChannelsWithAlphaInvokeLabtoBGRSrcP4(32f);
ForAllChannelsWithAlphaInvokeLabtoBGRInplace(32f);
} // namespace mpp::image::cuda
