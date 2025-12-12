#include "../convert_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeConvert(8s, 8u);
ForAllChannelsWithAlphaInvokeConvert(8s, 16u);
ForAllChannelsWithAlphaInvokeConvert(8s, 16s);
ForAllChannelsWithAlphaInvokeConvert(8s, 32u);
ForAllChannelsWithAlphaInvokeConvert(8s, 32s);
ForAllChannelsWithAlphaInvokeConvert(8s, 16bf);
ForAllChannelsWithAlphaInvokeConvert(8s, 16f);
ForAllChannelsWithAlphaInvokeConvert(8s, 32f);
ForAllChannelsWithAlphaInvokeConvert(8s, 64f);

} // namespace mpp::image::cuda
