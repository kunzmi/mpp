#if MPP_ENABLE_CUDA_BACKEND

#include "../mulMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeMulSrcSrcMask(32sc);
ForAllChannelsNoAlphaInvokeMulSrcSrcScaleMask(32sc);
ForAllChannelsNoAlphaInvokeMulSrcCMask(32sc);
ForAllChannelsNoAlphaInvokeMulSrcCScaleMask(32sc);
ForAllChannelsNoAlphaInvokeMulSrcDevCMask(32sc);
ForAllChannelsNoAlphaInvokeMulSrcDevCScaleMask(32sc);
ForAllChannelsNoAlphaInvokeMulInplaceSrcMask(32sc);
ForAllChannelsNoAlphaInvokeMulInplaceSrcScaleMask(32sc);
ForAllChannelsNoAlphaInvokeMulInplaceCMask(32sc);
ForAllChannelsNoAlphaInvokeMulInplaceCScaleMask(32sc);
ForAllChannelsNoAlphaInvokeMulInplaceDevCMask(32sc);
ForAllChannelsNoAlphaInvokeMulInplaceDevCScaleMask(32sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
