#if MPP_ENABLE_CUDA_BACKEND

#include "../compareEqEps_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeCompareEqEpsSrcSrc(32fc);
ForAllChannelsNoAlphaInvokeCompareEqEpsSrcC(32fc);
ForAllChannelsNoAlphaInvokeCompareEqEpsSrcDevC(32fc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
