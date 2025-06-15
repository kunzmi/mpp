#if OPP_ENABLE_CUDA_BACKEND

#include "../resize_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(64f);
InstantiateP2_ForGeomType(64f);
InstantiateP3_ForGeomType(64f);
InstantiateP4_ForGeomType(64f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
