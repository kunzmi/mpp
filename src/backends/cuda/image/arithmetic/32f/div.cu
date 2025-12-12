#include "../div_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrc(32f);
ForAllChannelsWithAlphaInvokeDivSrcC(32f);
ForAllChannelsWithAlphaInvokeDivSrcDevC(32f);
ForAllChannelsWithAlphaInvokeDivInplaceSrc(32f);
ForAllChannelsWithAlphaInvokeDivInplaceC(32f);
ForAllChannelsWithAlphaInvokeDivInplaceDevC(32f);
ForAllChannelsWithAlphaInvokeDivInvInplaceSrc(32f);
ForAllChannelsWithAlphaInvokeDivInvInplaceC(32f);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevC(32f);

} // namespace mpp::image::cuda
