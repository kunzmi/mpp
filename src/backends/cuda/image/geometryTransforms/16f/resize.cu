#if OPP_ENABLE_CUDA_BACKEND

#include "../resize_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16f);
InstantiateP2_ForGeomType(16f);
InstantiateP3_ForGeomType(16f);
InstantiateP4_ForGeomType(16f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
