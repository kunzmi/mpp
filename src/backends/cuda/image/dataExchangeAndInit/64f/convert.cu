#if MPP_ENABLE_CUDA_BACKEND

#include "../convert_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeConvert(64f, 16bf);
ForAllChannelsWithAlphaInvokeConvert(64f, 16f);
ForAllChannelsWithAlphaInvokeConvert(64f, 32f);
ForAllChannelsWithAlphaInvokeConvertRound(64f, 8u);
ForAllChannelsWithAlphaInvokeConvertRound(64f, 8s);
ForAllChannelsWithAlphaInvokeConvertRound(64f, 16u);
ForAllChannelsWithAlphaInvokeConvertRound(64f, 16s);
ForAllChannelsWithAlphaInvokeConvertRound(64f, 32u);
ForAllChannelsWithAlphaInvokeConvertRound(64f, 32s);
ForAllChannelsWithAlphaInvokeConvertScaleRound(64f, 8u);
ForAllChannelsWithAlphaInvokeConvertScaleRound(64f, 8s);
ForAllChannelsWithAlphaInvokeConvertScaleRound(64f, 16u);
ForAllChannelsWithAlphaInvokeConvertScaleRound(64f, 16s);
ForAllChannelsWithAlphaInvokeConvertScaleRound(64f, 32u);
ForAllChannelsWithAlphaInvokeConvertScaleRound(64f, 32s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
