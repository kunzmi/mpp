#include "../cfaToRgb_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeCfaToRgb3CSrc(32u);
ForAllChannelsWithAlphaInvokeCfaToRgb4CSrc(32u);

} // namespace mpp::image::cuda
