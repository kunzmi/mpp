#if OPP_ENABLE_CUDA_BACKEND

#include "../copy_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeCopy(32s);
ForAllChannelsNoAlphaInvokeCopy1Channel(32s);
ForAllChannelsNoAlphaInvokeCopyChannel1(32s);
ForAllChannelsNoAlphaInvokeCopyChannel(32s);
ForAllChannelsNoAlphaInvokeCopyPlanar2(32s);
ForAllChannelsNoAlphaInvokeCopyPlanar3(32s);
ForAllChannelsNoAlphaInvokeCopyPlanar4(32s);
ForAllChannelsNoAlphaInvokeCopy2Planar(32s);
ForAllChannelsNoAlphaInvokeCopy3Planar(32s);
ForAllChannelsNoAlphaInvokeCopy4Planar(32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
