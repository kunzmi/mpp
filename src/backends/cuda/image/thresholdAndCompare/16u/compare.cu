#if OPP_ENABLE_CUDA_BACKEND

#include "../compare_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeCompareSrcSrc(16u);
ForAllChannelsWithAlphaInvokeCompareSrcC(16u);
ForAllChannelsWithAlphaInvokeCompareSrcDevC(16u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
