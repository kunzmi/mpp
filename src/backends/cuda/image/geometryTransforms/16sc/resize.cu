#if OPP_ENABLE_CUDA_BACKEND

#include "../resize_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlpha(16sc);
InstantiateP2_ForGeomType(16sc);
InstantiateP3_ForGeomType(16sc);
InstantiateP4_ForGeomType(16sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
