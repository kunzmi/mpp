#if OPP_ENABLE_CUDA_BACKEND

#include "../copy_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeCopy(32sc);
ForAllChannelsNoAlphaInvokeCopy1Channel(32sc);
ForAllChannelsNoAlphaInvokeCopyChannel1(32sc);
ForAllChannelsNoAlphaInvokeCopyChannel(32sc);
ForAllChannelsNoAlphaInvokeCopyPlanar2(32sc);
ForAllChannelsNoAlphaInvokeCopyPlanar3(32sc);
ForAllChannelsNoAlphaInvokeCopyPlanar4(32sc);
ForAllChannelsNoAlphaInvokeCopy2Planar(32sc);
ForAllChannelsNoAlphaInvokeCopy3Planar(32sc);
ForAllChannelsNoAlphaInvokeCopy4Planar(32sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
