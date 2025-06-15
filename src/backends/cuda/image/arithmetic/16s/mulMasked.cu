#if OPP_ENABLE_CUDA_BACKEND

#include "../mulMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrcMask(16s);
ForAllChannelsWithAlphaInvokeMulSrcSrcScaleMask(16s);
ForAllChannelsWithAlphaInvokeMulSrcCMask(16s);
ForAllChannelsWithAlphaInvokeMulSrcCScaleMask(16s);
ForAllChannelsWithAlphaInvokeMulSrcDevCMask(16s);
ForAllChannelsWithAlphaInvokeMulSrcDevCScaleMask(16s);
ForAllChannelsWithAlphaInvokeMulInplaceSrcMask(16s);
ForAllChannelsWithAlphaInvokeMulInplaceSrcScaleMask(16s);
ForAllChannelsWithAlphaInvokeMulInplaceCMask(16s);
ForAllChannelsWithAlphaInvokeMulInplaceCScaleMask(16s);
ForAllChannelsWithAlphaInvokeMulInplaceDevCMask(16s);
ForAllChannelsWithAlphaInvokeMulInplaceDevCScaleMask(16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
