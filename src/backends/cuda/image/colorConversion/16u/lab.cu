#include "../lab_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRGBtoLabSrc(16u);
ForAllChannelsWithAlphaInvokeRGBtoLabP3Src(16u);
ForAllChannelsWithAlphaInvokeRGBtoLabP4Src(16u);
ForAllChannelsWithAlphaInvokeRGBtoLabP3SrcP3(16u);
ForAllChannelsWithAlphaInvokeRGBtoLabSrcP3(16u);
ForAllChannelsWithAlphaInvokeRGBtoLabSrcP4(16u);
ForAllChannelsWithAlphaInvokeRGBtoLabInplace(16u);

ForAllChannelsWithAlphaInvokeLabtoRGBSrc(16u);
ForAllChannelsWithAlphaInvokeLabtoRGBP3Src(16u);
ForAllChannelsWithAlphaInvokeLabtoRGBP4Src(16u);
ForAllChannelsWithAlphaInvokeLabtoRGBP3SrcP3(16u);
ForAllChannelsWithAlphaInvokeLabtoRGBSrcP3(16u);
ForAllChannelsWithAlphaInvokeLabtoRGBSrcP4(16u);
ForAllChannelsWithAlphaInvokeLabtoRGBInplace(16u);

ForAllChannelsWithAlphaInvokeBGRtoLabSrc(16u);
ForAllChannelsWithAlphaInvokeBGRtoLabP3Src(16u);
ForAllChannelsWithAlphaInvokeBGRtoLabP4Src(16u);
ForAllChannelsWithAlphaInvokeBGRtoLabP3SrcP3(16u);
ForAllChannelsWithAlphaInvokeBGRtoLabSrcP3(16u);
ForAllChannelsWithAlphaInvokeBGRtoLabSrcP4(16u);
ForAllChannelsWithAlphaInvokeBGRtoLabInplace(16u);

ForAllChannelsWithAlphaInvokeLabtoBGRSrc(16u);
ForAllChannelsWithAlphaInvokeLabtoBGRP3Src(16u);
ForAllChannelsWithAlphaInvokeLabtoBGRP4Src(16u);
ForAllChannelsWithAlphaInvokeLabtoBGRP3SrcP3(16u);
ForAllChannelsWithAlphaInvokeLabtoBGRSrcP3(16u);
ForAllChannelsWithAlphaInvokeLabtoBGRSrcP4(16u);
ForAllChannelsWithAlphaInvokeLabtoBGRInplace(16u);
} // namespace mpp::image::cuda
