#if OPP_ENABLE_CUDA_BACKEND

#include "../copy_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeCopy(32f);
ForAllChannelsNoAlphaInvokeCopy1Channel(32f);
ForAllChannelsNoAlphaInvokeCopyChannel1(32f);
ForAllChannelsNoAlphaInvokeCopyChannel(32f);
ForAllChannelsNoAlphaInvokeCopyPlanar2(32f);
ForAllChannelsNoAlphaInvokeCopyPlanar3(32f);
ForAllChannelsNoAlphaInvokeCopyPlanar4(32f);
ForAllChannelsNoAlphaInvokeCopy2Planar(32f);
ForAllChannelsNoAlphaInvokeCopy3Planar(32f);
ForAllChannelsNoAlphaInvokeCopy4Planar(32f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
