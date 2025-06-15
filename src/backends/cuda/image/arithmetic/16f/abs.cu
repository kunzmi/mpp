#if OPP_ENABLE_CUDA_BACKEND

#include "../abs_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAbsSrc(16f);
ForAllChannelsWithAlphaInvokeAbsInplace(16f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
