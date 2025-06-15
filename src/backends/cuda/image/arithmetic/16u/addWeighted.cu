#if OPP_ENABLE_CUDA_BACKEND

#include "../addWeighted_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddWeightedSrcSrc(16u);
ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrc(16u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
