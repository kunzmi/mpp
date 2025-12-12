#include "../sqrt_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSqrtSrc(16u);
ForAllChannelsWithAlphaInvokeSqrtInplace(16u);

} // namespace mpp::image::cuda
