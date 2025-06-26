#if MPP_ENABLE_CUDA_BACKEND

#include "../perspectiveBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(16sc);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(16sc);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(16sc);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(16sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
