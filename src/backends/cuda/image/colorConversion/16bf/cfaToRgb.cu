#include "../cfaToRgb_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeCfaToRgb3CSrc(16bf);
ForAllChannelsWithAlphaInvokeCfaToRgb4CSrc(16bf);

} // namespace mpp::image::cuda
