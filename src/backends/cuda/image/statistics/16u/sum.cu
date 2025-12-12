#include "../sum_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16u, 1);
ForAllChannelsWithAlpha(16u, 2);

} // namespace mpp::image::cuda
