#if OPP_ENABLE_CUDA_BACKEND

#include "../compare_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeCompareSrcSrc(16sc);
ForAllChannelsNoAlphaInvokeCompareSrcC(16sc);
ForAllChannelsNoAlphaInvokeCompareSrcDevC(16sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
