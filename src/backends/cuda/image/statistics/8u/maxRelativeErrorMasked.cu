#if OPP_ENABLE_CUDA_BACKEND

#include "../maxRelativeErrorMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(8u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
