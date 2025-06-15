#if OPP_ENABLE_CUDA_BACKEND

#include "../alphaComp_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlpha(16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
