#if MPP_ENABLE_CUDA_BACKEND

#include "../divMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrcScaleMask(8s);
ForAllChannelsWithAlphaInvokeDivSrcCScaleMask(8s);
ForAllChannelsWithAlphaInvokeDivSrcDevCScaleMask(8s);
ForAllChannelsWithAlphaInvokeDivInplaceSrcScaleMask(8s);
ForAllChannelsWithAlphaInvokeDivInplaceCScaleMask(8s);
ForAllChannelsWithAlphaInvokeDivInplaceDevCScaleMask(8s);
ForAllChannelsWithAlphaInvokeDivInvInplaceSrcScaleMask(8s);
ForAllChannelsWithAlphaInvokeDivInvInplaceCScaleMask(8s);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevCScaleMask(8s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
