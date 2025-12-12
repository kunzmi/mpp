#include "../sum_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(16sc, 1);
ForAllChannelsNoAlpha(16sc, 2);

} // namespace mpp::image::cuda
