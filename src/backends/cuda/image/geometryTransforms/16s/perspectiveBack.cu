#if OPP_ENABLE_CUDA_BACKEND

#include "../perspectiveBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16s);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(16s);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(16s);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
