#if OPP_ENABLE_CUDA_BACKEND

#include "../addWeighted_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddWeightedSrcSrc(16bf);
ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrc(16bf);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
