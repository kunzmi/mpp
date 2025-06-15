#if OPP_ENABLE_CUDA_BACKEND

#include "../lowpass_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
