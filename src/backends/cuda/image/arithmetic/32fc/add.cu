#include "../add_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeAddSrcSrc(32fc);
ForAllChannelsNoAlphaInvokeAddSrcC(32fc);
ForAllChannelsNoAlphaInvokeAddSrcDevC(32fc);
ForAllChannelsNoAlphaInvokeAddInplaceSrc(32fc);
ForAllChannelsNoAlphaInvokeAddInplaceC(32fc);
ForAllChannelsNoAlphaInvokeAddInplaceDevC(32fc);

} // namespace mpp::image::cuda
