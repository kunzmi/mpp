#if OPP_ENABLE_CUDA_BACKEND

#include "../perspectiveBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16u);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(16u);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(16u);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(16u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
