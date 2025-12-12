#include "../copy_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
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

} // namespace mpp::image::cuda
