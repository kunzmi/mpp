#if OPP_ENABLE_CUDA_BACKEND

#include "../averageErrorMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
