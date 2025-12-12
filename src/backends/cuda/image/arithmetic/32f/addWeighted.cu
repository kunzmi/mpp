#include "../addWeighted_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddWeightedSrcSrc(32f);
ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrc(32f);

} // namespace mpp::image::cuda
