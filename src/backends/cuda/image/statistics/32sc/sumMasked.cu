#if MPP_ENABLE_CUDA_BACKEND

#include "../sumMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(32sc, 1);
ForAllChannelsNoAlpha(32sc, 2);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
