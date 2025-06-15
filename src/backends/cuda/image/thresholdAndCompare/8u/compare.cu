#if OPP_ENABLE_CUDA_BACKEND

#include "../compare_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeCompareSrcSrc(8u);
ForAllChannelsWithAlphaInvokeCompareSrcC(8u);
ForAllChannelsWithAlphaInvokeCompareSrcDevC(8u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
