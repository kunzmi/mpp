#if MPP_ENABLE_CUDA_BACKEND

#include "../mirror_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMirrorSrc(32f);
ForAllChannelsWithAlphaInvokeMirrorInplace(32f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
