#if OPP_ENABLE_CUDA_BACKEND

#include "../resize_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16u);
InstantiateP2_ForGeomType(16u);
InstantiateP3_ForGeomType(16u);
InstantiateP4_ForGeomType(16u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
