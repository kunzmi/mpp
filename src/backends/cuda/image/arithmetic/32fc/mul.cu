#if OPP_ENABLE_CUDA_BACKEND

#include "../mul_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeMulSrcSrc(32fc);
ForAllChannelsNoAlphaInvokeMulSrcC(32fc);
ForAllChannelsNoAlphaInvokeMulSrcDevC(32fc);
ForAllChannelsNoAlphaInvokeMulInplaceSrc(32fc);
ForAllChannelsNoAlphaInvokeMulInplaceC(32fc);
ForAllChannelsNoAlphaInvokeMulInplaceDevC(32fc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
