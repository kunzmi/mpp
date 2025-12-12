#include "../cannyEdgeMaxSupression_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16s, 8u);

} // namespace mpp::image::cuda
