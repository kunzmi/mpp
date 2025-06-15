#if OPP_ENABLE_CUDA_BACKEND

#include "../divMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeDivSrcSrcScaleMask(32sc);
ForAllChannelsNoAlphaInvokeDivSrcCScaleMask(32sc);
ForAllChannelsNoAlphaInvokeDivSrcDevCScaleMask(32sc);
ForAllChannelsNoAlphaInvokeDivInplaceSrcScaleMask(32sc);
ForAllChannelsNoAlphaInvokeDivInplaceCScaleMask(32sc);
ForAllChannelsNoAlphaInvokeDivInplaceDevCScaleMask(32sc);
ForAllChannelsNoAlphaInvokeDivInvInplaceSrcScaleMask(32sc);
ForAllChannelsNoAlphaInvokeDivInvInplaceCScaleMask(32sc);
ForAllChannelsNoAlphaInvokeDivInvInplaceDevCScaleMask(32sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
