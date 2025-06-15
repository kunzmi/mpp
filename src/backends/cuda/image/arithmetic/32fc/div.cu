#if OPP_ENABLE_CUDA_BACKEND

#include "../div_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeDivSrcSrc(32fc);
ForAllChannelsNoAlphaInvokeDivSrcC(32fc);
ForAllChannelsNoAlphaInvokeDivSrcDevC(32fc);
ForAllChannelsNoAlphaInvokeDivInplaceSrc(32fc);
ForAllChannelsNoAlphaInvokeDivInplaceC(32fc);
ForAllChannelsNoAlphaInvokeDivInplaceDevC(32fc);
ForAllChannelsNoAlphaInvokeDivInvInplaceSrc(32fc);
ForAllChannelsNoAlphaInvokeDivInvInplaceC(32fc);
ForAllChannelsNoAlphaInvokeDivInvInplaceDevC(32fc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
