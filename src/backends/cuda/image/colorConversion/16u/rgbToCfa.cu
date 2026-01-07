#include "../rgbToCfa_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRgbToCfaSrc(16u);

} // namespace mpp::image::cuda
