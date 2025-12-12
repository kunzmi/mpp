#include "../transpose_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(64s); // needed for integral and sqrIntegral

} // namespace mpp::image::cuda
