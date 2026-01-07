#include "../lab_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRGBtoLabSrc(8u);
ForAllChannelsWithAlphaInvokeRGBtoLabP3Src(8u);
ForAllChannelsWithAlphaInvokeRGBtoLabP4Src(8u);
ForAllChannelsWithAlphaInvokeRGBtoLabP3SrcP3(8u);
ForAllChannelsWithAlphaInvokeRGBtoLabSrcP3(8u);
ForAllChannelsWithAlphaInvokeRGBtoLabSrcP4(8u);
ForAllChannelsWithAlphaInvokeRGBtoLabInplace(8u);

ForAllChannelsWithAlphaInvokeLabtoRGBSrc(8u);
ForAllChannelsWithAlphaInvokeLabtoRGBP3Src(8u);
ForAllChannelsWithAlphaInvokeLabtoRGBP4Src(8u);
ForAllChannelsWithAlphaInvokeLabtoRGBP3SrcP3(8u);
ForAllChannelsWithAlphaInvokeLabtoRGBSrcP3(8u);
ForAllChannelsWithAlphaInvokeLabtoRGBSrcP4(8u);
ForAllChannelsWithAlphaInvokeLabtoRGBInplace(8u);

ForAllChannelsWithAlphaInvokeBGRtoLabSrc(8u);
ForAllChannelsWithAlphaInvokeBGRtoLabP3Src(8u);
ForAllChannelsWithAlphaInvokeBGRtoLabP4Src(8u);
ForAllChannelsWithAlphaInvokeBGRtoLabP3SrcP3(8u);
ForAllChannelsWithAlphaInvokeBGRtoLabSrcP3(8u);
ForAllChannelsWithAlphaInvokeBGRtoLabSrcP4(8u);
ForAllChannelsWithAlphaInvokeBGRtoLabInplace(8u);

ForAllChannelsWithAlphaInvokeLabtoBGRSrc(8u);
ForAllChannelsWithAlphaInvokeLabtoBGRP3Src(8u);
ForAllChannelsWithAlphaInvokeLabtoBGRP4Src(8u);
ForAllChannelsWithAlphaInvokeLabtoBGRP3SrcP3(8u);
ForAllChannelsWithAlphaInvokeLabtoBGRSrcP3(8u);
ForAllChannelsWithAlphaInvokeLabtoBGRSrcP4(8u);
ForAllChannelsWithAlphaInvokeLabtoBGRInplace(8u);
} // namespace mpp::image::cuda
