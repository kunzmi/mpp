#include "../countInRange_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(8u, 64u, 64u);

} // namespace mpp::image::cuda
