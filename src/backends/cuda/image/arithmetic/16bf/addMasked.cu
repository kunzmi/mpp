#if OPP_ENABLE_CUDA_BACKEND

#include "../addMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrcMask(16bf);
ForAllChannelsWithAlphaInvokeAddSrcCMask(16bf);
ForAllChannelsWithAlphaInvokeAddSrcDevCMask(16bf);
ForAllChannelsWithAlphaInvokeAddInplaceSrcMask(16bf);
ForAllChannelsWithAlphaInvokeAddInplaceCMask(16bf);
ForAllChannelsWithAlphaInvokeAddInplaceDevCMask(16bf);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
