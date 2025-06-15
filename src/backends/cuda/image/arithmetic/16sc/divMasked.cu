#if OPP_ENABLE_CUDA_BACKEND

#include "../divMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeDivSrcSrcScaleMask(16sc);
ForAllChannelsNoAlphaInvokeDivSrcCScaleMask(16sc);
ForAllChannelsNoAlphaInvokeDivSrcDevCScaleMask(16sc);
ForAllChannelsNoAlphaInvokeDivInplaceSrcScaleMask(16sc);
ForAllChannelsNoAlphaInvokeDivInplaceCScaleMask(16sc);
ForAllChannelsNoAlphaInvokeDivInplaceDevCScaleMask(16sc);
ForAllChannelsNoAlphaInvokeDivInvInplaceSrcScaleMask(16sc);
ForAllChannelsNoAlphaInvokeDivInvInplaceCScaleMask(16sc);
ForAllChannelsNoAlphaInvokeDivInvInplaceDevCScaleMask(16sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
