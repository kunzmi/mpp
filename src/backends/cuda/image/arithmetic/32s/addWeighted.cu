#include "../addWeighted_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddWeightedSrcSrc(32s);
ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrc(32s);

} // namespace mpp::image::cuda
