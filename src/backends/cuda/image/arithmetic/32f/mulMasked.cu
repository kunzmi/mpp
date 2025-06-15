#if OPP_ENABLE_CUDA_BACKEND

#include "../mulMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMulSrcSrcMask(32f);
ForAllChannelsWithAlphaInvokeMulSrcCMask(32f);
ForAllChannelsWithAlphaInvokeMulSrcDevCMask(32f);
ForAllChannelsWithAlphaInvokeMulInplaceSrcMask(32f);
ForAllChannelsWithAlphaInvokeMulInplaceCMask(32f);
ForAllChannelsWithAlphaInvokeMulInplaceDevCMask(32f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
