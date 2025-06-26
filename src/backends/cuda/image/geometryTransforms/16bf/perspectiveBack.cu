#if MPP_ENABLE_CUDA_BACKEND

#include "../perspectiveBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16bf);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(16bf);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(16bf);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(16bf);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
