#include "../abs_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAbsSrc(16s);
ForAllChannelsWithAlphaInvokeAbsInplace(16s);

} // namespace mpp::image::cuda
