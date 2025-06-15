#if OPP_ENABLE_CUDA_BACKEND

#include "../boxFilter_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(32u, 32u);
ForAllChannelsWithAlpha(32u, 32f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
