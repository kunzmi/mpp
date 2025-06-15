#if OPP_ENABLE_CUDA_BACKEND

#include "../perspectiveBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(64f);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(64f);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(64f);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(64f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
