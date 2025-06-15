#if OPP_ENABLE_CUDA_BACKEND

#include "../not_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeNotSrc(32s);
ForAllChannelsWithAlphaInvokeNotInplace(32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
