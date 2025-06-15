#if OPP_ENABLE_CUDA_BACKEND

#include "../perspectiveBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(8u);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(8u);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(8u);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(8u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
