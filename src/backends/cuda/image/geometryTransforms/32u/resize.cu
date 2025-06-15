#if OPP_ENABLE_CUDA_BACKEND

#include "../resize_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(32u);
InstantiateP2_ForGeomType(32u);
InstantiateP3_ForGeomType(32u);
InstantiateP4_ForGeomType(32u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
