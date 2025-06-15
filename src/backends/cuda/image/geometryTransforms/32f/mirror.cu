#if OPP_ENABLE_CUDA_BACKEND

#include "../mirror_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeMirrorSrc(32f);
ForAllChannelsWithAlphaInvokeMirrorInplace(32f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
