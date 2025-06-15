#if OPP_ENABLE_CUDA_BACKEND

#include "../convert_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeConvert(16sc, 32sc);
ForAllChannelsNoAlphaInvokeConvert(16sc, 32fc);
ForAllChannelsNoAlphaInvokeConvert(16sc, 64fc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
