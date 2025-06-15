#if OPP_ENABLE_CUDA_BACKEND

#include "../copy_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeCopy(8s);
ForAllChannelsNoAlphaInvokeCopy1Channel(8s);
ForAllChannelsNoAlphaInvokeCopyChannel1(8s);
ForAllChannelsNoAlphaInvokeCopyChannel(8s);
ForAllChannelsNoAlphaInvokeCopyPlanar2(8s);
ForAllChannelsNoAlphaInvokeCopyPlanar3(8s);
ForAllChannelsNoAlphaInvokeCopyPlanar4(8s);
ForAllChannelsNoAlphaInvokeCopy2Planar(8s);
ForAllChannelsNoAlphaInvokeCopy3Planar(8s);
ForAllChannelsNoAlphaInvokeCopy4Planar(8s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
