#include "../boxFilter_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(8s, 8s);
ForAllChannelsWithAlpha(8s, 32f);

} // namespace mpp::image::cuda
