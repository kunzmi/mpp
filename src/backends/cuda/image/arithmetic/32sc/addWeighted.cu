#if OPP_ENABLE_CUDA_BACKEND

#include "../addWeighted_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeAddWeightedSrcSrc(32sc);
ForAllChannelsNoAlphaInvokeAddWeightedInplaceSrc(32sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
