#if OPP_ENABLE_CUDA_BACKEND

#include "../and_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAndSrcSrc(32s);
ForAllChannelsWithAlphaInvokeAndSrcC(32s);
ForAllChannelsWithAlphaInvokeAndSrcDevC(32s);
ForAllChannelsWithAlphaInvokeAndInplaceSrc(32s);
ForAllChannelsWithAlphaInvokeAndInplaceC(32s);
ForAllChannelsWithAlphaInvokeAndInplaceDevC(32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
