#if OPP_ENABLE_CUDA_BACKEND

#include "../and_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAndSrcSrc(8s);
ForAllChannelsWithAlphaInvokeAndSrcC(8s);
ForAllChannelsWithAlphaInvokeAndSrcDevC(8s);
ForAllChannelsWithAlphaInvokeAndInplaceSrc(8s);
ForAllChannelsWithAlphaInvokeAndInplaceC(8s);
ForAllChannelsWithAlphaInvokeAndInplaceDevC(8s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
