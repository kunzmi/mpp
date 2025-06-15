#if OPP_ENABLE_CUDA_BACKEND

#include "../convert_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeConvert(16bf, 32f);
ForAllChannelsWithAlphaInvokeConvert(16bf, 64f);
ForAllChannelsWithAlphaInvokeConvertRound(16bf, 8u);
ForAllChannelsWithAlphaInvokeConvertRound(16bf, 8s);
ForAllChannelsWithAlphaInvokeConvertRound(16bf, 16u);
ForAllChannelsWithAlphaInvokeConvertRound(16bf, 16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
