#if OPP_ENABLE_CUDA_BACKEND

#include "../resize_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(32s);
InstantiateP2_ForGeomType(32s);
InstantiateP3_ForGeomType(32s);
InstantiateP4_ForGeomType(32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
