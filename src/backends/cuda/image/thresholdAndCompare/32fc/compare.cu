#if OPP_ENABLE_CUDA_BACKEND

#include "../compare_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeCompareSrcSrc(32fc);
ForAllChannelsNoAlphaInvokeCompareSrcC(32fc);
ForAllChannelsNoAlphaInvokeCompareSrcDevC(32fc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
