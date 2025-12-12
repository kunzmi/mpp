#include "../minMaxEvery_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMinEverySrcSrc(32s);
ForAllChannelsWithAlphaInvokeMinEveryInplaceSrc(32s);
ForAllChannelsWithAlphaInvokeMaxEverySrcSrc(32s);
ForAllChannelsWithAlphaInvokeMaxEveryInplaceSrc(32s);

} // namespace mpp::image::cuda
