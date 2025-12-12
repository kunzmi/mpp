#include "../add_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrc(64f);
ForAllChannelsWithAlphaInvokeAddSrcC(64f);
ForAllChannelsWithAlphaInvokeAddSrcDevC(64f);
ForAllChannelsWithAlphaInvokeAddInplaceSrc(64f);
ForAllChannelsWithAlphaInvokeAddInplaceC(64f);
ForAllChannelsWithAlphaInvokeAddInplaceDevC(64f);

} // namespace mpp::image::cuda
