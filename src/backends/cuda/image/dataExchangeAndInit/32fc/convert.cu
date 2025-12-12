#include "../convert_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeConvertRound(32fc, 16sc);
ForAllChannelsNoAlphaInvokeConvertRound(32fc, 32sc);
ForAllChannelsNoAlphaInvokeConvertScaleRound(32fc, 16sc);
ForAllChannelsNoAlphaInvokeConvertScaleRound(32fc, 32sc);

} // namespace mpp::image::cuda
