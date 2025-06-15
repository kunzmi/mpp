#if OPP_ENABLE_CUDA_BACKEND

#include "../sumMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(32f, 1);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
