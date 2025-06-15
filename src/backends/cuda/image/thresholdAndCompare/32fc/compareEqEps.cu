#if OPP_ENABLE_CUDA_BACKEND

#include "../compareEqEps_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeCompareEqEpsSrcSrc(32fc);
ForAllChannelsNoAlphaInvokeCompareEqEpsSrcC(32fc);
ForAllChannelsNoAlphaInvokeCompareEqEpsSrcDevC(32fc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
