#if MPP_ENABLE_CUDA_BACKEND

#include "../copy_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeCopy(8u);
ForAllChannelsNoAlphaInvokeCopy1Channel(8u);
ForAllChannelsNoAlphaInvokeCopyChannel1(8u);
ForAllChannelsNoAlphaInvokeCopyChannel(8u);
ForAllChannelsNoAlphaInvokeCopyPlanar2(8u);
ForAllChannelsNoAlphaInvokeCopyPlanar3(8u);
ForAllChannelsNoAlphaInvokeCopyPlanar4(8u);
ForAllChannelsNoAlphaInvokeCopy2Planar(8u);
ForAllChannelsNoAlphaInvokeCopy3Planar(8u);
ForAllChannelsNoAlphaInvokeCopy4Planar(8u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
