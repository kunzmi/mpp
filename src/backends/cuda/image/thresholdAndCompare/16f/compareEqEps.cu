#if OPP_ENABLE_CUDA_BACKEND

#include "../compareEqEps_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeCompareEqEpsSrcSrc(16f);
ForAllChannelsWithAlphaInvokeCompareEqEpsSrcC(16f);
ForAllChannelsWithAlphaInvokeCompareEqEpsSrcDevC(16f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
