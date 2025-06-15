#if OPP_ENABLE_CUDA_BACKEND

#include "../exp_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeExpSrc(32u);
ForAllChannelsWithAlphaInvokeExpInplace(32u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
