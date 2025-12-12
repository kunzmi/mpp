#include "../exp_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeExpSrc(8s);
ForAllChannelsWithAlphaInvokeExpInplace(8s);

} // namespace mpp::image::cuda
