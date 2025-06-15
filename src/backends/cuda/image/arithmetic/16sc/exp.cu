#if OPP_ENABLE_CUDA_BACKEND

#include "../exp_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeExpSrc(16sc);
ForAllChannelsNoAlphaInvokeExpInplace(16sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
