#if OPP_ENABLE_CUDA_BACKEND

#include "../dilation_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(8s, 8s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
