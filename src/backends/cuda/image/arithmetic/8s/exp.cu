#if OPP_ENABLE_CUDA_BACKEND

#include "../exp_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeExpSrc(8s);
ForAllChannelsWithAlphaInvokeExpInplace(8s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
