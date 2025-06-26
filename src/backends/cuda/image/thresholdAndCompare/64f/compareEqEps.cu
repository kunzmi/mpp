#if MPP_ENABLE_CUDA_BACKEND

#include "../compareEqEps_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeCompareEqEpsSrcSrc(64f);
ForAllChannelsWithAlphaInvokeCompareEqEpsSrcC(64f);
ForAllChannelsWithAlphaInvokeCompareEqEpsSrcDevC(64f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
