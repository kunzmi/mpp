#if OPP_ENABLE_CUDA_BACKEND

#include "../exp_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeExpSrc(16f);
ForAllChannelsWithAlphaInvokeExpInplace(16f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
