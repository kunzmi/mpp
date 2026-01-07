#include "../cfaToRgb_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeCfaToRgb3CSrc(16s);
ForAllChannelsWithAlphaInvokeCfaToRgb4CSrc(16s);

} // namespace mpp::image::cuda
