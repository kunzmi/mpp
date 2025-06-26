#if MPP_ENABLE_CUDA_BACKEND

#include "../copy_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
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

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
