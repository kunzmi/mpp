#if OPP_ENABLE_CUDA_BACKEND

#include "../and_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAndSrcSrc(32u);
ForAllChannelsWithAlphaInvokeAndSrcC(32u);
ForAllChannelsWithAlphaInvokeAndSrcDevC(32u);
ForAllChannelsWithAlphaInvokeAndInplaceSrc(32u);
ForAllChannelsWithAlphaInvokeAndInplaceC(32u);
ForAllChannelsWithAlphaInvokeAndInplaceDevC(32u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
