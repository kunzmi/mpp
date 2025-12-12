#include "../sqrt_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSqrtSrc(32fc);
ForAllChannelsNoAlphaInvokeSqrtInplace(32fc);

} // namespace mpp::image::cuda
