#include "../cfaToRgb_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeCfaToRgb3CSrc(32f);
ForAllChannelsWithAlphaInvokeCfaToRgb4CSrc(32f);

} // namespace mpp::image::cuda
