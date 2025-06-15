#if OPP_ENABLE_CUDA_BACKEND

#include "../copy_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeCopy(16sc);
ForAllChannelsNoAlphaInvokeCopy1Channel(16sc);
ForAllChannelsNoAlphaInvokeCopyChannel1(16sc);
ForAllChannelsNoAlphaInvokeCopyChannel(16sc);
ForAllChannelsNoAlphaInvokeCopyPlanar2(16sc);
ForAllChannelsNoAlphaInvokeCopyPlanar3(16sc);
ForAllChannelsNoAlphaInvokeCopyPlanar4(16sc);
ForAllChannelsNoAlphaInvokeCopy2Planar(16sc);
ForAllChannelsNoAlphaInvokeCopy3Planar(16sc);
ForAllChannelsNoAlphaInvokeCopy4Planar(16sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
