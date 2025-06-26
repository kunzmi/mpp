#if MPP_ENABLE_CUDA_BACKEND

#include "../divMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrcScaleMask(32s);
ForAllChannelsWithAlphaInvokeDivSrcCScaleMask(32s);
ForAllChannelsWithAlphaInvokeDivSrcDevCScaleMask(32s);
ForAllChannelsWithAlphaInvokeDivInplaceSrcScaleMask(32s);
ForAllChannelsWithAlphaInvokeDivInplaceCScaleMask(32s);
ForAllChannelsWithAlphaInvokeDivInplaceDevCScaleMask(32s);
ForAllChannelsWithAlphaInvokeDivInvInplaceSrcScaleMask(32s);
ForAllChannelsWithAlphaInvokeDivInvInplaceCScaleMask(32s);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevCScaleMask(32s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
