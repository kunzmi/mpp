#if OPP_ENABLE_CUDA_BACKEND

#include "../addWeightedMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeAddWeightedSrcSrcMask(32sc);
ForAllChannelsNoAlphaInvokeAddWeightedInplaceSrcMask(32sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
