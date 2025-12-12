#include "../scharrHoriz_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16u, 16u);
ForAllChannelsWithAlpha(16u, 32s);

} // namespace mpp::image::cuda
