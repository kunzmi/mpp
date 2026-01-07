#include "../lab_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRGBtoLabSrc(16f);
ForAllChannelsWithAlphaInvokeRGBtoLabP3Src(16f);
ForAllChannelsWithAlphaInvokeRGBtoLabP4Src(16f);
ForAllChannelsWithAlphaInvokeRGBtoLabP3SrcP3(16f);
ForAllChannelsWithAlphaInvokeRGBtoLabSrcP3(16f);
ForAllChannelsWithAlphaInvokeRGBtoLabSrcP4(16f);
ForAllChannelsWithAlphaInvokeRGBtoLabInplace(16f);

ForAllChannelsWithAlphaInvokeLabtoRGBSrc(16f);
ForAllChannelsWithAlphaInvokeLabtoRGBP3Src(16f);
ForAllChannelsWithAlphaInvokeLabtoRGBP4Src(16f);
ForAllChannelsWithAlphaInvokeLabtoRGBP3SrcP3(16f);
ForAllChannelsWithAlphaInvokeLabtoRGBSrcP3(16f);
ForAllChannelsWithAlphaInvokeLabtoRGBSrcP4(16f);
ForAllChannelsWithAlphaInvokeLabtoRGBInplace(16f);

ForAllChannelsWithAlphaInvokeBGRtoLabSrc(16f);
ForAllChannelsWithAlphaInvokeBGRtoLabP3Src(16f);
ForAllChannelsWithAlphaInvokeBGRtoLabP4Src(16f);
ForAllChannelsWithAlphaInvokeBGRtoLabP3SrcP3(16f);
ForAllChannelsWithAlphaInvokeBGRtoLabSrcP3(16f);
ForAllChannelsWithAlphaInvokeBGRtoLabSrcP4(16f);
ForAllChannelsWithAlphaInvokeBGRtoLabInplace(16f);

ForAllChannelsWithAlphaInvokeLabtoBGRSrc(16f);
ForAllChannelsWithAlphaInvokeLabtoBGRP3Src(16f);
ForAllChannelsWithAlphaInvokeLabtoBGRP4Src(16f);
ForAllChannelsWithAlphaInvokeLabtoBGRP3SrcP3(16f);
ForAllChannelsWithAlphaInvokeLabtoBGRSrcP3(16f);
ForAllChannelsWithAlphaInvokeLabtoBGRSrcP4(16f);
ForAllChannelsWithAlphaInvokeLabtoBGRInplace(16f);
} // namespace mpp::image::cuda
