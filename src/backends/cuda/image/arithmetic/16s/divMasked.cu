#if OPP_ENABLE_CUDA_BACKEND

#include "../divMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeDivSrcSrcScaleMask(16s);
ForAllChannelsWithAlphaInvokeDivSrcCScaleMask(16s);
ForAllChannelsWithAlphaInvokeDivSrcDevCScaleMask(16s);
ForAllChannelsWithAlphaInvokeDivInplaceSrcScaleMask(16s);
ForAllChannelsWithAlphaInvokeDivInplaceCScaleMask(16s);
ForAllChannelsWithAlphaInvokeDivInplaceDevCScaleMask(16s);
ForAllChannelsWithAlphaInvokeDivInvInplaceSrcScaleMask(16s);
ForAllChannelsWithAlphaInvokeDivInvInplaceCScaleMask(16s);
ForAllChannelsWithAlphaInvokeDivInvInplaceDevCScaleMask(16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
