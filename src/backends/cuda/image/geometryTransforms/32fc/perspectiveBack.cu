#if OPP_ENABLE_CUDA_BACKEND

#include "../perspectiveBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlpha(32fc);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(32fc);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(32fc);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(32fc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
