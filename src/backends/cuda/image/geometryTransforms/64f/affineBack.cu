#if MPP_ENABLE_CUDA_BACKEND

#include "../affineBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(64f);
InstantiateInvokeAffineBackSrcP2ForGeomType(64f);
InstantiateInvokeAffineBackSrcP3ForGeomType(64f);
InstantiateInvokeAffineBackSrcP4ForGeomType(64f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
