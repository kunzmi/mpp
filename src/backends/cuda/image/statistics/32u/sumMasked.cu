#include "../sumMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(32u, 1);
ForAllChannelsWithAlpha(32u, 2);

} // namespace mpp::image::cuda
