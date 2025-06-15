#if OPP_ENABLE_CUDA_BACKEND

#include "../sharpen_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlpha(32sc, 32sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
