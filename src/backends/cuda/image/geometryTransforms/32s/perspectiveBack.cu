#if OPP_ENABLE_CUDA_BACKEND

#include "../perspectiveBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(32s);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(32s);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(32s);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
