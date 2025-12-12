#include "../sqr_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSqrSrc(8u);
ForAllChannelsWithAlphaInvokeSqrInplace(8u);

} // namespace mpp::image::cuda
