#include "../add_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrc(32f);
ForAllChannelsWithAlphaInvokeAddSrcC(32f);
ForAllChannelsWithAlphaInvokeAddSrcDevC(32f);
ForAllChannelsWithAlphaInvokeAddInplaceSrc(32f);
ForAllChannelsWithAlphaInvokeAddInplaceC(32f);
ForAllChannelsWithAlphaInvokeAddInplaceDevC(32f);

} // namespace mpp::image::cuda
