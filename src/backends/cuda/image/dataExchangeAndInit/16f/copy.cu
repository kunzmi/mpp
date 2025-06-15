#if OPP_ENABLE_CUDA_BACKEND

#include "../copy_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeCopy(16f);
ForAllChannelsNoAlphaInvokeCopy1Channel(16f);
ForAllChannelsNoAlphaInvokeCopyChannel1(16f);
ForAllChannelsNoAlphaInvokeCopyChannel(16f);
ForAllChannelsNoAlphaInvokeCopyPlanar2(16f);
ForAllChannelsNoAlphaInvokeCopyPlanar3(16f);
ForAllChannelsNoAlphaInvokeCopyPlanar4(16f);
ForAllChannelsNoAlphaInvokeCopy2Planar(16f);
ForAllChannelsNoAlphaInvokeCopy3Planar(16f);
ForAllChannelsNoAlphaInvokeCopy4Planar(16f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
