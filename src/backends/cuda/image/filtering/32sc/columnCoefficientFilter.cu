#if OPP_ENABLE_CUDA_BACKEND

#include "../columnCoefficientFilter_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlpha(32sc, float);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
