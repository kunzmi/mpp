#if OPP_ENABLE_CUDA_BACKEND

#include "../gaussFixed_impl.h"

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
