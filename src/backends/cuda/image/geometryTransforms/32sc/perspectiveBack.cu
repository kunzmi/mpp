#if OPP_ENABLE_CUDA_BACKEND

#include "../perspectiveBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlpha(32sc);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(32sc);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(32sc);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(32sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
