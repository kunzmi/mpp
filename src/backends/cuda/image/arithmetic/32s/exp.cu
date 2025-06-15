#if OPP_ENABLE_CUDA_BACKEND

#include "../exp_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeExpSrc(32s);
ForAllChannelsWithAlphaInvokeExpInplace(32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
