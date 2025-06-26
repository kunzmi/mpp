#if MPP_ENABLE_CUDA_BACKEND

#include "../mirror_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMirrorSrc(16f);
ForAllChannelsWithAlphaInvokeMirrorInplace(16f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
