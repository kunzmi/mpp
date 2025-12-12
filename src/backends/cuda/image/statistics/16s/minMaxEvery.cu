#include "../minMaxEvery_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMinEverySrcSrc(16s);
ForAllChannelsWithAlphaInvokeMinEveryInplaceSrc(16s);
ForAllChannelsWithAlphaInvokeMaxEverySrcSrc(16s);
ForAllChannelsWithAlphaInvokeMaxEveryInplaceSrc(16s);

} // namespace mpp::image::cuda
