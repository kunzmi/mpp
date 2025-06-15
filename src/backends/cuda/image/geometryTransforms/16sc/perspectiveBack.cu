#if OPP_ENABLE_CUDA_BACKEND

#include "../perspectiveBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlpha(16sc);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(16sc);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(16sc);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(16sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
