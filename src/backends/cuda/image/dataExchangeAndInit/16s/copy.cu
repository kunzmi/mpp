#if OPP_ENABLE_CUDA_BACKEND

#include "../copy_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeCopy(16s);
ForAllChannelsNoAlphaInvokeCopy1Channel(16s);
ForAllChannelsNoAlphaInvokeCopyChannel1(16s);
ForAllChannelsNoAlphaInvokeCopyChannel(16s);
ForAllChannelsNoAlphaInvokeCopyPlanar2(16s);
ForAllChannelsNoAlphaInvokeCopyPlanar3(16s);
ForAllChannelsNoAlphaInvokeCopyPlanar4(16s);
ForAllChannelsNoAlphaInvokeCopy2Planar(16s);
ForAllChannelsNoAlphaInvokeCopy3Planar(16s);
ForAllChannelsNoAlphaInvokeCopy4Planar(16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
