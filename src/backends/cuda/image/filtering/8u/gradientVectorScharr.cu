#if OPP_ENABLE_CUDA_BACKEND

#include "../gradientVectorScharr_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(8u, 16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
