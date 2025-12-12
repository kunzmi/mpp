#include "../convert_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeConvert(16sc, 32sc);
ForAllChannelsNoAlphaInvokeConvert(16sc, 32fc);

} // namespace mpp::image::cuda
