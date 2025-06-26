#if MPP_ENABLE_CUDA_BACKEND

#include "../perspectiveBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(32sc);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(32sc);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(32sc);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(32sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
