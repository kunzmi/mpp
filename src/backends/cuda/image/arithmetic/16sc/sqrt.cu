#include "../sqrt_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSqrtSrc(16sc);
ForAllChannelsNoAlphaInvokeSqrtInplace(16sc);

} // namespace mpp::image::cuda
