#if MPP_ENABLE_CUDA_BACKEND

#include "../divMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrcScaleMask(32u);
ForAllChannelsWithAlphaInvokeDivSrcCScaleMask(32u);
ForAllChannelsWithAlphaInvokeDivSrcDevCScaleMask(32u);
ForAllChannelsWithAlphaInvokeDivInplaceSrcScaleMask(32u);
ForAllChannelsWithAlphaInvokeDivInplaceCScaleMask(32u);
ForAllChannelsWithAlphaInvokeDivInplaceDevCScaleMask(32u);
ForAllChannelsWithAlphaInvokeDivInvInplaceSrcScaleMask(32u);
ForAllChannelsWithAlphaInvokeDivInvInplaceCScaleMask(32u);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevCScaleMask(32u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
