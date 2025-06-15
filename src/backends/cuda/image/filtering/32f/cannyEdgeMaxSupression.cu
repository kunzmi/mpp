#if OPP_ENABLE_CUDA_BACKEND

#include "../cannyEdgeMaxSupression_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(32f, 8u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
