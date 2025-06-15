#if OPP_ENABLE_CUDA_BACKEND

#include "../divMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
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

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
