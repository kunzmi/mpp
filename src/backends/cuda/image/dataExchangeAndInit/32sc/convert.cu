#if OPP_ENABLE_CUDA_BACKEND

#include "../convert_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeConvertScaleRound(32sc, 16sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
