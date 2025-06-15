#if OPP_ENABLE_CUDA_BACKEND

#include "../add_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeAddSrcSrc(32fc);
ForAllChannelsNoAlphaInvokeAddSrcC(32fc);
ForAllChannelsNoAlphaInvokeAddSrcDevC(32fc);
ForAllChannelsNoAlphaInvokeAddInplaceSrc(32fc);
ForAllChannelsNoAlphaInvokeAddInplaceC(32fc);
ForAllChannelsNoAlphaInvokeAddInplaceDevC(32fc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
