#if OPP_ENABLE_CUDA_BACKEND

#include "../affineBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16u);
InstantiateInvokeAffineBackSrcP2ForGeomType(16u);
InstantiateInvokeAffineBackSrcP3ForGeomType(16u);
InstantiateInvokeAffineBackSrcP4ForGeomType(16u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
