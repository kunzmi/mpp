#if OPP_ENABLE_CUDA_BACKEND

#include "../addProductMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlpha(32fc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
