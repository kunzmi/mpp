#include "../cfaToRgb_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeCfaToRgb3CSrc(8u);
ForAllChannelsWithAlphaInvokeCfaToRgb4CSrc(8u);

} // namespace mpp::image::cuda
