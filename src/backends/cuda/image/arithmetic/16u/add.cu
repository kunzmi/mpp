#include "../add_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrc(16u);
ForAllChannelsWithAlphaInvokeAddSrcSrcScale(16u);
ForAllChannelsWithAlphaInvokeAddSrcC(16u);
ForAllChannelsWithAlphaInvokeAddSrcCScale(16u);
ForAllChannelsWithAlphaInvokeAddSrcDevC(16u);
ForAllChannelsWithAlphaInvokeAddSrcDevCScale(16u);
ForAllChannelsWithAlphaInvokeAddInplaceSrc(16u);
ForAllChannelsWithAlphaInvokeAddInplaceSrcScale(16u);
ForAllChannelsWithAlphaInvokeAddInplaceC(16u);
ForAllChannelsWithAlphaInvokeAddInplaceCScale(16u);
ForAllChannelsWithAlphaInvokeAddInplaceDevC(16u);
ForAllChannelsWithAlphaInvokeAddInplaceDevCScale(16u);

} // namespace mpp::image::cuda
