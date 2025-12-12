#include "../div_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrcScale(8s);
ForAllChannelsWithAlphaInvokeDivSrcCScale(8s);
ForAllChannelsWithAlphaInvokeDivSrcDevCScale(8s);
ForAllChannelsWithAlphaInvokeDivInplaceSrcScale(8s);
ForAllChannelsWithAlphaInvokeDivInplaceCScale(8s);
ForAllChannelsWithAlphaInvokeDivInplaceDevCScale(8s);
ForAllChannelsWithAlphaInvokeDivInvInplaceSrcScale(8s);
ForAllChannelsWithAlphaInvokeDivInvInplaceCScale(8s);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevCScale(8s);

} // namespace mpp::image::cuda
