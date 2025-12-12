#include "../convert_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeConvert(16f, 32f);
ForAllChannelsWithAlphaInvokeConvert(16f, 64f);
ForAllChannelsWithAlphaInvokeConvertRound(16f, 8u);
ForAllChannelsWithAlphaInvokeConvertRound(16f, 8s);
ForAllChannelsWithAlphaInvokeConvertRound(16f, 16u);
ForAllChannelsWithAlphaInvokeConvertRound(16f, 16s);
ForAllChannelsWithAlphaInvokeConvertRound(16f, 32u);
ForAllChannelsWithAlphaInvokeConvertRound(16f, 32s);

} // namespace mpp::image::cuda
