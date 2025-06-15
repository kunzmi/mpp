#if OPP_ENABLE_CUDA_BACKEND

#include "../fixedSizeSeparableFilter_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16f, float);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
