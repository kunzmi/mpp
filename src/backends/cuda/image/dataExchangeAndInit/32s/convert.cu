#if OPP_ENABLE_CUDA_BACKEND

#include "../convert_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeConvert(32s, 8u);
ForAllChannelsWithAlphaInvokeConvert(32s, 8s);
ForAllChannelsWithAlphaInvokeConvert(32s, 32u);
ForAllChannelsWithAlphaInvokeConvert(32s, 16bf);
ForAllChannelsWithAlphaInvokeConvert(32s, 16f);
ForAllChannelsWithAlphaInvokeConvert(32s, 32f);
ForAllChannelsWithAlphaInvokeConvert(32s, 64f);
ForAllChannelsWithAlphaInvokeConvertScaleRound(32s, 16u);
ForAllChannelsWithAlphaInvokeConvertScaleRound(32s, 16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
