#if OPP_ENABLE_CUDA_BACKEND

#include "../mulMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrcMask(64f);
ForAllChannelsWithAlphaInvokeMulSrcCMask(64f);
ForAllChannelsWithAlphaInvokeMulSrcDevCMask(64f);
ForAllChannelsWithAlphaInvokeMulInplaceSrcMask(64f);
ForAllChannelsWithAlphaInvokeMulInplaceCMask(64f);
ForAllChannelsWithAlphaInvokeMulInplaceDevCMask(64f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
