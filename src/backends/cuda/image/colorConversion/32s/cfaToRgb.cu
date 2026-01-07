#include "../cfaToRgb_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeCfaToRgb3CSrc(32s);
ForAllChannelsWithAlphaInvokeCfaToRgb4CSrc(32s);

} // namespace mpp::image::cuda
