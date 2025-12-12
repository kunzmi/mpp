#include "../div_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrcScale(16s);
ForAllChannelsWithAlphaInvokeDivSrcCScale(16s);
ForAllChannelsWithAlphaInvokeDivSrcDevCScale(16s);
ForAllChannelsWithAlphaInvokeDivInplaceSrcScale(16s);
ForAllChannelsWithAlphaInvokeDivInplaceCScale(16s);
ForAllChannelsWithAlphaInvokeDivInplaceDevCScale(16s);
ForAllChannelsWithAlphaInvokeDivInvInplaceSrcScale(16s);
ForAllChannelsWithAlphaInvokeDivInvInplaceCScale(16s);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevCScale(16s);

} // namespace mpp::image::cuda
