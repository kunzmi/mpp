#if OPP_ENABLE_CUDA_BACKEND

#include "../compareEqEps_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeCompareEqEpsSrcSrc(32f);
ForAllChannelsWithAlphaInvokeCompareEqEpsSrcC(32f);
ForAllChannelsWithAlphaInvokeCompareEqEpsSrcDevC(32f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
