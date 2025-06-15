#if OPP_ENABLE_CUDA_BACKEND

#include "../compare_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeCompareSrcSrc(32sc);
ForAllChannelsNoAlphaInvokeCompareSrcC(32sc);
ForAllChannelsNoAlphaInvokeCompareSrcDevC(32sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
