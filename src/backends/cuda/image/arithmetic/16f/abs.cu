#include "../abs_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAbsSrc(16f);
ForAllChannelsWithAlphaInvokeAbsInplace(16f);

} // namespace mpp::image::cuda
