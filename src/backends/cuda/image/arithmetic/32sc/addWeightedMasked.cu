#include "../addWeightedMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeAddWeightedSrcSrcMask(32sc);
ForAllChannelsNoAlphaInvokeAddWeightedInplaceSrcMask(32sc);

} // namespace mpp::image::cuda
