#if OPP_ENABLE_CUDA_BACKEND

#include "../copy_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeCopy(16bf);
ForAllChannelsNoAlphaInvokeCopy1Channel(16bf);
ForAllChannelsNoAlphaInvokeCopyChannel1(16bf);
ForAllChannelsNoAlphaInvokeCopyChannel(16bf);
ForAllChannelsNoAlphaInvokeCopyPlanar2(16bf);
ForAllChannelsNoAlphaInvokeCopyPlanar3(16bf);
ForAllChannelsNoAlphaInvokeCopyPlanar4(16bf);
ForAllChannelsNoAlphaInvokeCopy2Planar(16bf);
ForAllChannelsNoAlphaInvokeCopy3Planar(16bf);
ForAllChannelsNoAlphaInvokeCopy4Planar(16bf);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
