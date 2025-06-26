#if MPP_ENABLE_CUDA_BACKEND

#include "../mirror_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMirrorSrc(16bf);
ForAllChannelsWithAlphaInvokeMirrorInplace(16bf);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
