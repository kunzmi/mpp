#include "../add_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrc(32s);
ForAllChannelsWithAlphaInvokeAddSrcSrcScale(32s);
ForAllChannelsWithAlphaInvokeAddSrcC(32s);
ForAllChannelsWithAlphaInvokeAddSrcCScale(32s);
ForAllChannelsWithAlphaInvokeAddSrcDevC(32s);
ForAllChannelsWithAlphaInvokeAddSrcDevCScale(32s);
ForAllChannelsWithAlphaInvokeAddInplaceSrc(32s);
ForAllChannelsWithAlphaInvokeAddInplaceSrcScale(32s);
ForAllChannelsWithAlphaInvokeAddInplaceC(32s);
ForAllChannelsWithAlphaInvokeAddInplaceCScale(32s);
ForAllChannelsWithAlphaInvokeAddInplaceDevC(32s);
ForAllChannelsWithAlphaInvokeAddInplaceDevCScale(32s);

} // namespace mpp::image::cuda
