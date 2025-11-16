#if MPP_ENABLE_CUDA_BACKEND

#include "../convert_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeConvert(8u, 16u);
ForAllChannelsWithAlphaInvokeConvert(8u, 16s);
ForAllChannelsWithAlphaInvokeConvert(8u, 32u);
ForAllChannelsWithAlphaInvokeConvert(8u, 32s);
ForAllChannelsWithAlphaInvokeConvert(8u, 16bf);
ForAllChannelsWithAlphaInvokeConvert(8u, 16f);
ForAllChannelsWithAlphaInvokeConvert(8u, 32f);
ForAllChannelsWithAlphaInvokeConvert(8u, 64f);
ForAllChannelsWithAlphaInvokeConvertScaleRound(8u, 8s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
