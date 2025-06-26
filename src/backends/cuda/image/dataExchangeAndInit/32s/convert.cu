#if MPP_ENABLE_CUDA_BACKEND

#include "../convert_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
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

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
