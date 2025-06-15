#if OPP_ENABLE_CUDA_BACKEND

#include "../mulMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrcMask(32u);
ForAllChannelsWithAlphaInvokeMulSrcSrcScaleMask(32u);
ForAllChannelsWithAlphaInvokeMulSrcCMask(32u);
ForAllChannelsWithAlphaInvokeMulSrcCScaleMask(32u);
ForAllChannelsWithAlphaInvokeMulSrcDevCMask(32u);
ForAllChannelsWithAlphaInvokeMulSrcDevCScaleMask(32u);
ForAllChannelsWithAlphaInvokeMulInplaceSrcMask(32u);
ForAllChannelsWithAlphaInvokeMulInplaceSrcScaleMask(32u);
ForAllChannelsWithAlphaInvokeMulInplaceCMask(32u);
ForAllChannelsWithAlphaInvokeMulInplaceCScaleMask(32u);
ForAllChannelsWithAlphaInvokeMulInplaceDevCMask(32u);
ForAllChannelsWithAlphaInvokeMulInplaceDevCScaleMask(32u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
