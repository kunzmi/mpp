#if OPP_ENABLE_CUDA_BACKEND

#include "../resize_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(32f);
InstantiateP2_ForGeomType(32f);
InstantiateP3_ForGeomType(32f);
InstantiateP4_ForGeomType(32f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
