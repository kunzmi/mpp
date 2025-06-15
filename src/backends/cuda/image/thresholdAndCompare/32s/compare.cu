#if OPP_ENABLE_CUDA_BACKEND

#include "../compare_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeCompareSrcSrc(32s);
ForAllChannelsWithAlphaInvokeCompareSrcC(32s);
ForAllChannelsWithAlphaInvokeCompareSrcDevC(32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
