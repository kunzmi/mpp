#if OPP_ENABLE_CUDA_BACKEND

#include "../integral_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlpha(8s, 32s);
ForAllChannelsNoAlpha(8s, 32f);
ForAllChannelsNoAlpha(8s, 64s);
ForAllChannelsNoAlpha(8s, 64f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
