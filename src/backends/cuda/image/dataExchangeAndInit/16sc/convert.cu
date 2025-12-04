#if MPP_ENABLE_CUDA_BACKEND

#include "../convert_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeConvert(16sc, 32sc);
ForAllChannelsNoAlphaInvokeConvert(16sc, 32fc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
