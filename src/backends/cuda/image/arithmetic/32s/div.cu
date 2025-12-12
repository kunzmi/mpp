#include "../div_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrcScale(32s);
ForAllChannelsWithAlphaInvokeDivSrcCScale(32s);
ForAllChannelsWithAlphaInvokeDivSrcDevCScale(32s);
ForAllChannelsWithAlphaInvokeDivInplaceSrcScale(32s);
ForAllChannelsWithAlphaInvokeDivInplaceCScale(32s);
ForAllChannelsWithAlphaInvokeDivInplaceDevCScale(32s);
ForAllChannelsWithAlphaInvokeDivInvInplaceSrcScale(32s);
ForAllChannelsWithAlphaInvokeDivInvInplaceCScale(32s);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevCScale(32s);

} // namespace mpp::image::cuda
