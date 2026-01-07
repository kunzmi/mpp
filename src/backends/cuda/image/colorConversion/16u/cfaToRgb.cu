#include "../cfaToRgb_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeCfaToRgb3CSrc(16u);
ForAllChannelsWithAlphaInvokeCfaToRgb4CSrc(16u);

} // namespace mpp::image::cuda
