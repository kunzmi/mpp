#if MPP_ENABLE_CUDA_BACKEND

#include "../divMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrcScaleMask(8u);
ForAllChannelsWithAlphaInvokeDivSrcCScaleMask(8u);
ForAllChannelsWithAlphaInvokeDivSrcDevCScaleMask(8u);
ForAllChannelsWithAlphaInvokeDivInplaceSrcScaleMask(8u);
ForAllChannelsWithAlphaInvokeDivInplaceCScaleMask(8u);
ForAllChannelsWithAlphaInvokeDivInplaceDevCScaleMask(8u);
ForAllChannelsWithAlphaInvokeDivInvInplaceSrcScaleMask(8u);
ForAllChannelsWithAlphaInvokeDivInvInplaceCScaleMask(8u);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevCScaleMask(8u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
