#include "../sqrt_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSqrtSrc(64f);
ForAllChannelsWithAlphaInvokeSqrtInplace(64f);

} // namespace mpp::image::cuda
