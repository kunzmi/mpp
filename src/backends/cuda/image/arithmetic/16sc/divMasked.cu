#if MPP_ENABLE_CUDA_BACKEND

#include "../divMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeDivSrcSrcScaleMask(16sc);
ForAllChannelsNoAlphaInvokeDivSrcCScaleMask(16sc);
ForAllChannelsNoAlphaInvokeDivSrcDevCScaleMask(16sc);
ForAllChannelsNoAlphaInvokeDivInplaceSrcScaleMask(16sc);
ForAllChannelsNoAlphaInvokeDivInplaceCScaleMask(16sc);
ForAllChannelsNoAlphaInvokeDivInplaceDevCScaleMask(16sc);
ForAllChannelsNoAlphaInvokeDivInvInplaceSrcScaleMask(16sc);
ForAllChannelsNoAlphaInvokeDivInvInplaceCScaleMask(16sc);
ForAllChannelsNoAlphaInvokeDivInvInplaceDevCScaleMask(16sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
