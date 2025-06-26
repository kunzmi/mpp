#if MPP_ENABLE_CUDA_BACKEND

#include "../divMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrcScaleMask(16u);
ForAllChannelsWithAlphaInvokeDivSrcCScaleMask(16u);
ForAllChannelsWithAlphaInvokeDivSrcDevCScaleMask(16u);
ForAllChannelsWithAlphaInvokeDivInplaceSrcScaleMask(16u);
ForAllChannelsWithAlphaInvokeDivInplaceCScaleMask(16u);
ForAllChannelsWithAlphaInvokeDivInplaceDevCScaleMask(16u);
ForAllChannelsWithAlphaInvokeDivInvInplaceSrcScaleMask(16u);
ForAllChannelsWithAlphaInvokeDivInvInplaceCScaleMask(16u);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevCScaleMask(16u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
