#if OPP_ENABLE_CUDA_BACKEND

#include "../rowCoefficientFilter_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(32s, float);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
