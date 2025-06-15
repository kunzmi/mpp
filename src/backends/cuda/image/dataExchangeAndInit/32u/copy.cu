#if OPP_ENABLE_CUDA_BACKEND

#include "../copy_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeCopy(32u);
ForAllChannelsNoAlphaInvokeCopy1Channel(32u);
ForAllChannelsNoAlphaInvokeCopyChannel1(32u);
ForAllChannelsNoAlphaInvokeCopyChannel(32u);
ForAllChannelsNoAlphaInvokeCopyPlanar2(32u);
ForAllChannelsNoAlphaInvokeCopyPlanar3(32u);
ForAllChannelsNoAlphaInvokeCopyPlanar4(32u);
ForAllChannelsNoAlphaInvokeCopy2Planar(32u);
ForAllChannelsNoAlphaInvokeCopy3Planar(32u);
ForAllChannelsNoAlphaInvokeCopy4Planar(32u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
