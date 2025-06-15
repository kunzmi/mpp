#if OPP_ENABLE_CUDA_BACKEND

#include "../convert_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeConvertRound(32fc, 16sc);
ForAllChannelsNoAlphaInvokeConvertScaleRound(32fc, 16sc);
ForAllChannelsNoAlphaInvokeConvertScaleRound(32fc, 32sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
