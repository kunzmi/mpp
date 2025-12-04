#if MPP_ENABLE_CUDA_BACKEND

#include "../convert_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeConvert(16bf, 32f);
ForAllChannelsWithAlphaInvokeConvert(16bf, 64f);
ForAllChannelsWithAlphaInvokeConvertRound(16bf, 8u);
ForAllChannelsWithAlphaInvokeConvertRound(16bf, 8s);
ForAllChannelsWithAlphaInvokeConvertRound(16bf, 16u);
ForAllChannelsWithAlphaInvokeConvertRound(16bf, 16s);
ForAllChannelsWithAlphaInvokeConvertRound(16bf, 32u);
ForAllChannelsWithAlphaInvokeConvertRound(16bf, 32s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
