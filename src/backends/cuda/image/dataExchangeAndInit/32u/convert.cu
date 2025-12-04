#if MPP_ENABLE_CUDA_BACKEND

#include "../convert_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeConvert(32u, 8u);
ForAllChannelsWithAlphaInvokeConvert(32u, 8s);
ForAllChannelsWithAlphaInvokeConvert(32u, 16u);
ForAllChannelsWithAlphaInvokeConvert(32u, 16s);
ForAllChannelsWithAlphaInvokeConvert(32u, 32s);
ForAllChannelsWithAlphaInvokeConvert(32u, 16bf);
ForAllChannelsWithAlphaInvokeConvert(32u, 16f);
ForAllChannelsWithAlphaInvokeConvert(32u, 32f);
ForAllChannelsWithAlphaInvokeConvert(32u, 64f);
ForAllChannelsWithAlphaInvokeConvertScaleRound(32u, 8u);
ForAllChannelsWithAlphaInvokeConvertScaleRound(32u, 8s);
ForAllChannelsWithAlphaInvokeConvertScaleRound(32u, 16u);
ForAllChannelsWithAlphaInvokeConvertScaleRound(32u, 16s);
ForAllChannelsWithAlphaInvokeConvertScaleRound(32u, 32s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
