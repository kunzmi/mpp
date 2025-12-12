#include "../fixedSizeBoxFilter_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(8u, 8u);
ForAllChannelsWithAlpha(8u, 32f);

} // namespace mpp::image::cuda
