#if OPP_ENABLE_CUDA_BACKEND

#include "../sobelHoriz_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlpha(16sc, 16sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
