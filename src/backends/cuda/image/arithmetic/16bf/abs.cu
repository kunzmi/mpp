#include "../abs_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAbsSrc(16bf);
ForAllChannelsWithAlphaInvokeAbsInplace(16bf);

} // namespace mpp::image::cuda
