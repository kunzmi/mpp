#include "../div_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrcScale(8u);
ForAllChannelsWithAlphaInvokeDivSrcCScale(8u);
ForAllChannelsWithAlphaInvokeDivSrcDevCScale(8u);
ForAllChannelsWithAlphaInvokeDivInplaceSrcScale(8u);
ForAllChannelsWithAlphaInvokeDivInplaceCScale(8u);
ForAllChannelsWithAlphaInvokeDivInplaceDevCScale(8u);
ForAllChannelsWithAlphaInvokeDivInvInplaceSrcScale(8u);
ForAllChannelsWithAlphaInvokeDivInvInplaceCScale(8u);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevCScale(8u);

} // namespace mpp::image::cuda
