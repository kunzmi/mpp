#if OPP_ENABLE_CUDA_BACKEND

#include "../copy_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeCopy(64f);
ForAllChannelsNoAlphaInvokeCopy1Channel(64f);
ForAllChannelsNoAlphaInvokeCopyChannel1(64f);
ForAllChannelsNoAlphaInvokeCopyChannel(64f);
ForAllChannelsNoAlphaInvokeCopyPlanar2(64f);
ForAllChannelsNoAlphaInvokeCopyPlanar3(64f);
ForAllChannelsNoAlphaInvokeCopyPlanar4(64f);
ForAllChannelsNoAlphaInvokeCopy2Planar(64f);
ForAllChannelsNoAlphaInvokeCopy3Planar(64f);
ForAllChannelsNoAlphaInvokeCopy4Planar(64f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
