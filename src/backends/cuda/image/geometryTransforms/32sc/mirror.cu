#if MPP_ENABLE_CUDA_BACKEND

#include "../mirror_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeMirrorSrc(32sc);
ForAllChannelsNoAlphaInvokeMirrorInplace(32sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
