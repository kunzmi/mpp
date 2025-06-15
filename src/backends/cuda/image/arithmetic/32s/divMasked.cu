#if OPP_ENABLE_CUDA_BACKEND

#include "../divMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
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

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
