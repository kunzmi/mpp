#include "../rgbToCfa_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRgbToCfaSrc(32u);

} // namespace mpp::image::cuda
