#include "../cfaToRgb_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeCfaToRgb3CSrc(16f);
ForAllChannelsWithAlphaInvokeCfaToRgb4CSrc(16f);

} // namespace mpp::image::cuda
