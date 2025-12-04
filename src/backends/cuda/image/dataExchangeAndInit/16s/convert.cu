#if MPP_ENABLE_CUDA_BACKEND

#include "../convert_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeConvert(16s, 8u);
ForAllChannelsWithAlphaInvokeConvert(16s, 8s);
ForAllChannelsWithAlphaInvokeConvert(16s, 16u);
ForAllChannelsWithAlphaInvokeConvert(16s, 32u);
ForAllChannelsWithAlphaInvokeConvert(16s, 32s);
ForAllChannelsWithAlphaInvokeConvert(16s, 16bf);
ForAllChannelsWithAlphaInvokeConvert(16s, 16f);
ForAllChannelsWithAlphaInvokeConvert(16s, 32f);
ForAllChannelsWithAlphaInvokeConvert(16s, 64f);
ForAllChannelsWithAlphaInvokeConvertScaleRound(16s, 8s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
