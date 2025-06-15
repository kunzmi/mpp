#if OPP_ENABLE_CUDA_BACKEND

#include "../perspectiveBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16bf);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(16bf);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(16bf);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(16bf);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
