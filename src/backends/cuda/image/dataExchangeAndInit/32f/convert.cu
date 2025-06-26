#if MPP_ENABLE_CUDA_BACKEND

#include "../convert_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeConvert(32f, 16bf);
ForAllChannelsWithAlphaInvokeConvert(32f, 16f);
ForAllChannelsWithAlphaInvokeConvert(32f, 64f);
ForAllChannelsWithAlphaInvokeConvertRound(32f, 8u);
ForAllChannelsWithAlphaInvokeConvertRound(32f, 8s);
ForAllChannelsWithAlphaInvokeConvertRound(32f, 16u);
ForAllChannelsWithAlphaInvokeConvertRound(32f, 16s);
ForAllChannelsWithAlphaInvokeConvertRound(32f, 16f);
ForAllChannelsWithAlphaInvokeConvertRound(32f, 16bf);
ForAllChannelsWithAlphaInvokeConvertScaleRound(32f, 8u);
ForAllChannelsWithAlphaInvokeConvertScaleRound(32f, 8s);
ForAllChannelsWithAlphaInvokeConvertScaleRound(32f, 16u);
ForAllChannelsWithAlphaInvokeConvertScaleRound(32f, 16s);
ForAllChannelsWithAlphaInvokeConvertScaleRound(32f, 32u);
ForAllChannelsWithAlphaInvokeConvertScaleRound(32f, 32s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
