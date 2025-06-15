#if OPP_ENABLE_CUDA_BACKEND

#include "../mirror_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeMirrorSrc(32fc);
ForAllChannelsNoAlphaInvokeMirrorInplace(32fc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
