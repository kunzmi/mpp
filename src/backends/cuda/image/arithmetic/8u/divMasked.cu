#if OPP_ENABLE_CUDA_BACKEND

#include "../divMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
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

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
