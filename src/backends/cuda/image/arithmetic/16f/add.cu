#include "../add_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrc(16f);
ForAllChannelsWithAlphaInvokeAddSrcC(16f);
ForAllChannelsWithAlphaInvokeAddSrcDevC(16f);
ForAllChannelsWithAlphaInvokeAddInplaceSrc(16f);
ForAllChannelsWithAlphaInvokeAddInplaceC(16f);
ForAllChannelsWithAlphaInvokeAddInplaceDevC(16f);

} // namespace mpp::image::cuda
