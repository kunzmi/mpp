#if MPP_ENABLE_CUDA_BACKEND

#include "../mirror_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeMirrorSrc(16sc);
ForAllChannelsNoAlphaInvokeMirrorInplace(16sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
