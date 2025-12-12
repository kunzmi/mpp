#include "../countInRangeMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16u, 64u, 64u);

} // namespace mpp::image::cuda
