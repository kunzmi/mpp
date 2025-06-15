#if OPP_ENABLE_CUDA_BACKEND

#include "../compareEqEps_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeCompareEqEpsSrcSrc(16bf);
ForAllChannelsWithAlphaInvokeCompareEqEpsSrcC(16bf);
ForAllChannelsWithAlphaInvokeCompareEqEpsSrcDevC(16bf);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
