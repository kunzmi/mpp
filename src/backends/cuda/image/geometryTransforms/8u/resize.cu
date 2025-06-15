#if OPP_ENABLE_CUDA_BACKEND

#include "../resize_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(8u);
InstantiateP2_ForGeomType(8u);
InstantiateP3_ForGeomType(8u);
InstantiateP4_ForGeomType(8u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
