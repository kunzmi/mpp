#include "../sqr_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSqrSrc(16bf);
ForAllChannelsWithAlphaInvokeSqrInplace(16bf);

} // namespace mpp::image::cuda
