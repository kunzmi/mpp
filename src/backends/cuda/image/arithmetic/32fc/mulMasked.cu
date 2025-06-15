#if OPP_ENABLE_CUDA_BACKEND

#include "../mulMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeMulSrcSrcMask(32fc);
ForAllChannelsNoAlphaInvokeMulSrcCMask(32fc);
ForAllChannelsNoAlphaInvokeMulSrcDevCMask(32fc);
ForAllChannelsNoAlphaInvokeMulInplaceSrcMask(32fc);
ForAllChannelsNoAlphaInvokeMulInplaceCMask(32fc);
ForAllChannelsNoAlphaInvokeMulInplaceDevCMask(32fc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
