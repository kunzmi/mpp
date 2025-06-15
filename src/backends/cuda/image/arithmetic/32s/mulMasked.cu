#if OPP_ENABLE_CUDA_BACKEND

#include "../mulMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrcMask(32s);
ForAllChannelsWithAlphaInvokeMulSrcSrcScaleMask(32s);
ForAllChannelsWithAlphaInvokeMulSrcCMask(32s);
ForAllChannelsWithAlphaInvokeMulSrcCScaleMask(32s);
ForAllChannelsWithAlphaInvokeMulSrcDevCMask(32s);
ForAllChannelsWithAlphaInvokeMulSrcDevCScaleMask(32s);
ForAllChannelsWithAlphaInvokeMulInplaceSrcMask(32s);
ForAllChannelsWithAlphaInvokeMulInplaceSrcScaleMask(32s);
ForAllChannelsWithAlphaInvokeMulInplaceCMask(32s);
ForAllChannelsWithAlphaInvokeMulInplaceCScaleMask(32s);
ForAllChannelsWithAlphaInvokeMulInplaceDevCMask(32s);
ForAllChannelsWithAlphaInvokeMulInplaceDevCScaleMask(32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
