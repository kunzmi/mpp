#if OPP_ENABLE_CUDA_BACKEND

#include "../separableFilter_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(32f, float);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
