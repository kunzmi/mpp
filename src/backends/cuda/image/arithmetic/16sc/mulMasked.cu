#if MPP_ENABLE_CUDA_BACKEND

#include "../mulMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeMulSrcSrcMask(16sc);
ForAllChannelsNoAlphaInvokeMulSrcSrcScaleMask(16sc);
ForAllChannelsNoAlphaInvokeMulSrcCMask(16sc);
ForAllChannelsNoAlphaInvokeMulSrcCScaleMask(16sc);
ForAllChannelsNoAlphaInvokeMulSrcDevCMask(16sc);
ForAllChannelsNoAlphaInvokeMulSrcDevCScaleMask(16sc);
ForAllChannelsNoAlphaInvokeMulInplaceSrcMask(16sc);
ForAllChannelsNoAlphaInvokeMulInplaceSrcScaleMask(16sc);
ForAllChannelsNoAlphaInvokeMulInplaceCMask(16sc);
ForAllChannelsNoAlphaInvokeMulInplaceCScaleMask(16sc);
ForAllChannelsNoAlphaInvokeMulInplaceDevCMask(16sc);
ForAllChannelsNoAlphaInvokeMulInplaceDevCScaleMask(16sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
