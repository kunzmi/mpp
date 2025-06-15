#if OPP_ENABLE_CUDA_BACKEND

#include "../mulMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrcMask(16f);
ForAllChannelsWithAlphaInvokeMulSrcCMask(16f);
ForAllChannelsWithAlphaInvokeMulSrcDevCMask(16f);
ForAllChannelsWithAlphaInvokeMulInplaceSrcMask(16f);
ForAllChannelsWithAlphaInvokeMulInplaceCMask(16f);
ForAllChannelsWithAlphaInvokeMulInplaceDevCMask(16f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
