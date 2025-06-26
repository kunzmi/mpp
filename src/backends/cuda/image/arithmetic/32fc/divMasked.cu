#if MPP_ENABLE_CUDA_BACKEND

#include "../divMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeDivSrcSrcMask(32fc);
ForAllChannelsNoAlphaInvokeDivSrcCMask(32fc);
ForAllChannelsNoAlphaInvokeDivSrcDevCMask(32fc);
ForAllChannelsNoAlphaInvokeDivInplaceSrcMask(32fc);
ForAllChannelsNoAlphaInvokeDivInplaceCMask(32fc);
ForAllChannelsNoAlphaInvokeDivInplaceDevCMask(32fc);
ForAllChannelsNoAlphaInvokeDivInvInplaceSrcMask(32fc);
ForAllChannelsNoAlphaInvokeDivInvInplaceCMask(32fc);
ForAllChannelsNoAlphaInvokeDivInvInplaceDevCMask(32fc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
