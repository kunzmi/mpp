#if OPP_ENABLE_CUDA_BACKEND

#include "../mulMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrcMask(16u);
ForAllChannelsWithAlphaInvokeMulSrcSrcScaleMask(16u);
ForAllChannelsWithAlphaInvokeMulSrcCMask(16u);
ForAllChannelsWithAlphaInvokeMulSrcCScaleMask(16u);
ForAllChannelsWithAlphaInvokeMulSrcDevCMask(16u);
ForAllChannelsWithAlphaInvokeMulSrcDevCScaleMask(16u);
ForAllChannelsWithAlphaInvokeMulInplaceSrcMask(16u);
ForAllChannelsWithAlphaInvokeMulInplaceSrcScaleMask(16u);
ForAllChannelsWithAlphaInvokeMulInplaceCMask(16u);
ForAllChannelsWithAlphaInvokeMulInplaceCScaleMask(16u);
ForAllChannelsWithAlphaInvokeMulInplaceDevCMask(16u);
ForAllChannelsWithAlphaInvokeMulInplaceDevCScaleMask(16u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
