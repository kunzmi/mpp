#include "../convert_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeConvertScaleRound(32sc, 16sc);
ForAllChannelsNoAlphaInvokeConvert(32sc, 16sc);
ForAllChannelsNoAlphaInvokeConvert(32sc, 32fc);

} // namespace mpp::image::cuda
