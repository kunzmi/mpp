#if OPP_ENABLE_CUDA_BACKEND

#include "../convert_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeConvert(16u, 8u);
ForAllChannelsWithAlphaInvokeConvert(16u, 32u);
ForAllChannelsWithAlphaInvokeConvert(16u, 32s);
ForAllChannelsWithAlphaInvokeConvert(16u, 16bf);
ForAllChannelsWithAlphaInvokeConvert(16u, 16f);
ForAllChannelsWithAlphaInvokeConvert(16u, 32f);
ForAllChannelsWithAlphaInvokeConvert(16u, 64f);
ForAllChannelsWithAlphaInvokeConvertScaleRound(16u, 8s);
ForAllChannelsWithAlphaInvokeConvertScaleRound(16u, 16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
