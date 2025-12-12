#include "../addWeighted_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddWeightedSrcSrc(8s);
ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrc(8s);

} // namespace mpp::image::cuda
