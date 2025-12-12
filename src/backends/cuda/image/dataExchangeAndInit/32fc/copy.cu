#include "../copy_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeCopy(32fc);
ForAllChannelsNoAlphaInvokeCopy1Channel(32fc);
ForAllChannelsNoAlphaInvokeCopyChannel1(32fc);
ForAllChannelsNoAlphaInvokeCopyChannel(32fc);
ForAllChannelsNoAlphaInvokeCopyPlanar2(32fc);
ForAllChannelsNoAlphaInvokeCopyPlanar3(32fc);
ForAllChannelsNoAlphaInvokeCopyPlanar4(32fc);
ForAllChannelsNoAlphaInvokeCopy2Planar(32fc);
ForAllChannelsNoAlphaInvokeCopy3Planar(32fc);
ForAllChannelsNoAlphaInvokeCopy4Planar(32fc);

} // namespace mpp::image::cuda
