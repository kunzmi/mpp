#include "../sobelVert_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(8s, 8s);
ForAllChannelsWithAlpha(8s, 16s);

} // namespace mpp::image::cuda
