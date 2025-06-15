#if OPP_ENABLE_CUDA_BACKEND

#include "../addWeightedMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddWeightedSrcSrcMask(32s);
ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrcMask(32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
