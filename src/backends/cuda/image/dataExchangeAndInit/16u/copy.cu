#include "../copy_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeCopy(16u);
ForAllChannelsNoAlphaInvokeCopy1Channel(16u);
ForAllChannelsNoAlphaInvokeCopyChannel1(16u);
ForAllChannelsNoAlphaInvokeCopyChannel(16u);
ForAllChannelsNoAlphaInvokeCopyPlanar2(16u);
ForAllChannelsNoAlphaInvokeCopyPlanar3(16u);
ForAllChannelsNoAlphaInvokeCopyPlanar4(16u);
ForAllChannelsNoAlphaInvokeCopy2Planar(16u);
ForAllChannelsNoAlphaInvokeCopy3Planar(16u);
ForAllChannelsNoAlphaInvokeCopy4Planar(16u);

} // namespace mpp::image::cuda
