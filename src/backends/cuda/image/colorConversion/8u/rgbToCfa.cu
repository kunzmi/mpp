#include "../rgbToCfa_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRgbToCfaSrc(8u);

} // namespace mpp::image::cuda
