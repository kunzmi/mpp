#if OPP_ENABLE_CUDA_BACKEND

#include "../resize_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16s);
InstantiateP2_ForGeomType(16s);
InstantiateP3_ForGeomType(16s);
InstantiateP4_ForGeomType(16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
