#if OPP_ENABLE_CUDA_BACKEND

#include "../fixedSizeErosion_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(64f, 64f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
