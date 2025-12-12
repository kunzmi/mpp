#include "../sqrt_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSqrtSrc(16f);
ForAllChannelsWithAlphaInvokeSqrtInplace(16f);

} // namespace mpp::image::cuda
