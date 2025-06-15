#if OPP_ENABLE_CUDA_BACKEND

#include "../fixedSizeFilter_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16s, 16s);
ForAllChannelsWithAlpha(16s, 32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
