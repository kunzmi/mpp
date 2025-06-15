#if OPP_ENABLE_CUDA_BACKEND

#include "../not_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeNotSrc(16s);
ForAllChannelsWithAlphaInvokeNotInplace(16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
