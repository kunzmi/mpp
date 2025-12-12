#include "../add_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrc(16bf);
ForAllChannelsWithAlphaInvokeAddSrcC(16bf);
ForAllChannelsWithAlphaInvokeAddSrcDevC(16bf);
ForAllChannelsWithAlphaInvokeAddInplaceSrc(16bf);
ForAllChannelsWithAlphaInvokeAddInplaceC(16bf);
ForAllChannelsWithAlphaInvokeAddInplaceDevC(16bf);

} // namespace mpp::image::cuda
