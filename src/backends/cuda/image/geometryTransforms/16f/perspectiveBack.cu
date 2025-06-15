#if OPP_ENABLE_CUDA_BACKEND

#include "../perspectiveBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16f);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(16f);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(16f);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(16f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
