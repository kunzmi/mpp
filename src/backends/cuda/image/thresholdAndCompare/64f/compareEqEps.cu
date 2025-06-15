#if OPP_ENABLE_CUDA_BACKEND

#include "../compareEqEps_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeCompareEqEpsSrcSrc(64f);
ForAllChannelsWithAlphaInvokeCompareEqEpsSrcC(64f);
ForAllChannelsWithAlphaInvokeCompareEqEpsSrcDevC(64f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
