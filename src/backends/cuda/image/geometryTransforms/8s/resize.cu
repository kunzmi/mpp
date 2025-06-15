#if OPP_ENABLE_CUDA_BACKEND

#include "../resize_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(8s);
InstantiateP2_ForGeomType(8s);
InstantiateP3_ForGeomType(8s);
InstantiateP4_ForGeomType(8s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
