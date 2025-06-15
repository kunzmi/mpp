#if OPP_ENABLE_CUDA_BACKEND

#include "../div_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeDivSrcSrcScale(32sc);
ForAllChannelsNoAlphaInvokeDivSrcCScale(32sc);
ForAllChannelsNoAlphaInvokeDivSrcDevCScale(32sc);
ForAllChannelsNoAlphaInvokeDivInplaceSrcScale(32sc);
ForAllChannelsNoAlphaInvokeDivInplaceCScale(32sc);
ForAllChannelsNoAlphaInvokeDivInplaceDevCScale(32sc);
ForAllChannelsNoAlphaInvokeDivInvInplaceSrcScale(32sc);
ForAllChannelsNoAlphaInvokeDivInvInplaceCScale(32sc);
ForAllChannelsNoAlphaInvokeDivInvInplaceDevCScale(32sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
