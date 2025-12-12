#include "../addWeighted_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeAddWeightedSrcSrc(32sc);
ForAllChannelsNoAlphaInvokeAddWeightedInplaceSrc(32sc);

} // namespace mpp::image::cuda
