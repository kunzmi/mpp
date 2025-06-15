#if OPP_ENABLE_CUDA_BACKEND

#include "../and_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAndSrcSrc(16u);
ForAllChannelsWithAlphaInvokeAndSrcC(16u);
ForAllChannelsWithAlphaInvokeAndSrcDevC(16u);
ForAllChannelsWithAlphaInvokeAndInplaceSrc(16u);
ForAllChannelsWithAlphaInvokeAndInplaceC(16u);
ForAllChannelsWithAlphaInvokeAndInplaceDevC(16u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
