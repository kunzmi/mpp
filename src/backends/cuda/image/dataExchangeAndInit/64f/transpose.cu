#if OPP_ENABLE_CUDA_BACKEND

#include "../transpose_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlpha(64f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
