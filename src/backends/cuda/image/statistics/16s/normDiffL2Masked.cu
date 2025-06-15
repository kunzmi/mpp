#if OPP_ENABLE_CUDA_BACKEND

#include "../normDiffL2Masked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
