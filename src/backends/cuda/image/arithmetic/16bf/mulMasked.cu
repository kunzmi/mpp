#if OPP_ENABLE_CUDA_BACKEND

#include "../mulMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrcMask(16bf);
ForAllChannelsWithAlphaInvokeMulSrcCMask(16bf);
ForAllChannelsWithAlphaInvokeMulSrcDevCMask(16bf);
ForAllChannelsWithAlphaInvokeMulInplaceSrcMask(16bf);
ForAllChannelsWithAlphaInvokeMulInplaceCMask(16bf);
ForAllChannelsWithAlphaInvokeMulInplaceDevCMask(16bf);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
