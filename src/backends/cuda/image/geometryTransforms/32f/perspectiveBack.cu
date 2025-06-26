#if MPP_ENABLE_CUDA_BACKEND

#include "../perspectiveBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(32f);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(32f);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(32f);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(32f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
