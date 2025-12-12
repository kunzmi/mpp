#include "../abs_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAbsSrc(32f);
ForAllChannelsWithAlphaInvokeAbsInplace(32f);

} // namespace mpp::image::cuda
