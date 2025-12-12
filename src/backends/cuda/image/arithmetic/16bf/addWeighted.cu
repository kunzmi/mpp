#include "../addWeighted_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddWeightedSrcSrc(16bf);
ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrc(16bf);

} // namespace mpp::image::cuda
