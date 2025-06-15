#if OPP_ENABLE_CUDA_BACKEND

#include "../divMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
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

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
