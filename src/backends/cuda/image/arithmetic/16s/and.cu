#if OPP_ENABLE_CUDA_BACKEND

#include "../and_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAndSrcSrc(16s);
ForAllChannelsWithAlphaInvokeAndSrcC(16s);
ForAllChannelsWithAlphaInvokeAndSrcDevC(16s);
ForAllChannelsWithAlphaInvokeAndInplaceSrc(16s);
ForAllChannelsWithAlphaInvokeAndInplaceC(16s);
ForAllChannelsWithAlphaInvokeAndInplaceDevC(16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
