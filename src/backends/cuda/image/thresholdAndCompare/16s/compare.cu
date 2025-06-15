#if OPP_ENABLE_CUDA_BACKEND

#include "../compare_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeCompareSrcSrc(16s);
ForAllChannelsWithAlphaInvokeCompareSrcC(16s);
ForAllChannelsWithAlphaInvokeCompareSrcDevC(16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
