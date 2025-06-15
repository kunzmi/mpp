#if OPP_ENABLE_CUDA_BACKEND

#include "../addMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrcMask(32f);
ForAllChannelsWithAlphaInvokeAddSrcCMask(32f);
ForAllChannelsWithAlphaInvokeAddSrcDevCMask(32f);
ForAllChannelsWithAlphaInvokeAddInplaceSrcMask(32f);
ForAllChannelsWithAlphaInvokeAddInplaceCMask(32f);
ForAllChannelsWithAlphaInvokeAddInplaceDevCMask(32f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
