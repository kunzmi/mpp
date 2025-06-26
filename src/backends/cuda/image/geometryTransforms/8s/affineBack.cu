#if MPP_ENABLE_CUDA_BACKEND

#include "../affineBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(8s);
InstantiateInvokeAffineBackSrcP2ForGeomType(8s);
InstantiateInvokeAffineBackSrcP3ForGeomType(8s);
InstantiateInvokeAffineBackSrcP4ForGeomType(8s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
