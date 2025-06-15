#if OPP_ENABLE_CUDA_BACKEND

#include "../and_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAndSrcSrc(8u);
ForAllChannelsWithAlphaInvokeAndSrcC(8u);
ForAllChannelsWithAlphaInvokeAndSrcDevC(8u);
ForAllChannelsWithAlphaInvokeAndInplaceSrc(8u);
ForAllChannelsWithAlphaInvokeAndInplaceC(8u);
ForAllChannelsWithAlphaInvokeAndInplaceDevC(8u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
