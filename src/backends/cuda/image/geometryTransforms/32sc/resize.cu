#if OPP_ENABLE_CUDA_BACKEND

#include "../resize_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlpha(32sc);
InstantiateP2_ForGeomType(32sc);
InstantiateP3_ForGeomType(32sc);
InstantiateP4_ForGeomType(32sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
