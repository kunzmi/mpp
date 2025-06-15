#if OPP_ENABLE_CUDA_BACKEND

#include "../resize_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlpha(32fc);
InstantiateP2_ForGeomType(32fc);
InstantiateP3_ForGeomType(32fc);
InstantiateP4_ForGeomType(32fc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
