#if OPP_ENABLE_CUDA_BACKEND

#include "../mul_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeMulSrcSrc(32sc);
ForAllChannelsNoAlphaInvokeMulSrcSrcScale(32sc);
ForAllChannelsNoAlphaInvokeMulSrcC(32sc);
ForAllChannelsNoAlphaInvokeMulSrcCScale(32sc);
ForAllChannelsNoAlphaInvokeMulSrcDevC(32sc);
ForAllChannelsNoAlphaInvokeMulSrcDevCScale(32sc);
ForAllChannelsNoAlphaInvokeMulInplaceSrc(32sc);
ForAllChannelsNoAlphaInvokeMulInplaceSrcScale(32sc);
ForAllChannelsNoAlphaInvokeMulInplaceC(32sc);
ForAllChannelsNoAlphaInvokeMulInplaceCScale(32sc);
ForAllChannelsNoAlphaInvokeMulInplaceDevC(32sc);
ForAllChannelsNoAlphaInvokeMulInplaceDevCScale(32sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
