#if OPP_ENABLE_CUDA_BACKEND

#include "../convert_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeConvert(16f, 32f);
ForAllChannelsWithAlphaInvokeConvert(16f, 64f);
ForAllChannelsWithAlphaInvokeConvertRound(16f, 8u);
ForAllChannelsWithAlphaInvokeConvertRound(16f, 8s);
ForAllChannelsWithAlphaInvokeConvertRound(16f, 16u);
ForAllChannelsWithAlphaInvokeConvertRound(16f, 16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
