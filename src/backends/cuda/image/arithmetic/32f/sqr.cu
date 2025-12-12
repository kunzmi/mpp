#include "../sqr_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSqrSrc(32f);
ForAllChannelsWithAlphaInvokeSqrInplace(32f);

} // namespace mpp::image::cuda
