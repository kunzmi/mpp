#if OPP_ENABLE_CUDA_BACKEND

#include "../resize_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16bf);
InstantiateP2_ForGeomType(16bf);
InstantiateP3_ForGeomType(16bf);
InstantiateP4_ForGeomType(16bf);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
