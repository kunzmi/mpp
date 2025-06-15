#if OPP_ENABLE_CUDA_BACKEND

#include "../addWeightedMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddWeightedSrcSrcMask(8s);
ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrcMask(8s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
