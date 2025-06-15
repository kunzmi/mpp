#if OPP_ENABLE_CUDA_BACKEND

#include "../columnCoefficientFilter_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlpha(16sc, float);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
