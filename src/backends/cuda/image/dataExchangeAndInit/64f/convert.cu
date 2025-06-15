#if OPP_ENABLE_CUDA_BACKEND

#include "../convert_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeConvertRound(64f, 8u);
ForAllChannelsWithAlphaInvokeConvertRound(64f, 8s);
ForAllChannelsWithAlphaInvokeConvertRound(64f, 16u);
ForAllChannelsWithAlphaInvokeConvertRound(64f, 16s);
ForAllChannelsWithAlphaInvokeConvertScaleRound(64f, 8u);
ForAllChannelsWithAlphaInvokeConvertScaleRound(64f, 8s);
ForAllChannelsWithAlphaInvokeConvertScaleRound(64f, 16u);
ForAllChannelsWithAlphaInvokeConvertScaleRound(64f, 16s);
ForAllChannelsWithAlphaInvokeConvertScaleRound(64f, 32u);
ForAllChannelsWithAlphaInvokeConvertScaleRound(64f, 32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
