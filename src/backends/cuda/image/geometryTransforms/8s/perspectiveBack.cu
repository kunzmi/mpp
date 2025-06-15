#if OPP_ENABLE_CUDA_BACKEND

#include "../perspectiveBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(8s);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(8s);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(8s);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(8s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
