#if OPP_ENABLE_CUDA_BACKEND

#include "../compare_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeCompareSrcSrc(32u);
ForAllChannelsWithAlphaInvokeCompareSrcC(32u);
ForAllChannelsWithAlphaInvokeCompareSrcDevC(32u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
