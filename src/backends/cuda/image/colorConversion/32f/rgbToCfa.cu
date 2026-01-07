#include "../rgbToCfa_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeRgbToCfaSrc(32f);

} // namespace mpp::image::cuda
