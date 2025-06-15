#if OPP_ENABLE_CUDA_BACKEND

#include "../perspectiveBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(32u);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(32u);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(32u);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(32u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
