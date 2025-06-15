#if OPP_ENABLE_CUDA_BACKEND

#include "../not_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeNotSrc(8s);
ForAllChannelsWithAlphaInvokeNotInplace(8s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
