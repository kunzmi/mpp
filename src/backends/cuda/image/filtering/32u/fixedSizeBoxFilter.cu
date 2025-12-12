#include "../fixedSizeBoxFilter_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(32u, 32u);
ForAllChannelsWithAlpha(32u, 32f);

} // namespace mpp::image::cuda
