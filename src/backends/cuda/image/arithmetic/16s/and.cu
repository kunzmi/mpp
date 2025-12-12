#include "../and_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAndSrcSrc(16s);
ForAllChannelsWithAlphaInvokeAndSrcC(16s);
ForAllChannelsWithAlphaInvokeAndSrcDevC(16s);
ForAllChannelsWithAlphaInvokeAndInplaceSrc(16s);
ForAllChannelsWithAlphaInvokeAndInplaceC(16s);
ForAllChannelsWithAlphaInvokeAndInplaceDevC(16s);

} // namespace mpp::image::cuda
