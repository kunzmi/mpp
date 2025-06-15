#if OPP_ENABLE_CUDA_BACKEND

#include "../morphologyGradient_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(32s, 32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
