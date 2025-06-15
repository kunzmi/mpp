#if OPP_ENABLE_CUDA_BACKEND

#include "../perspectiveBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(32f);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(32f);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(32f);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(32f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
