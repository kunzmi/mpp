#if OPP_ENABLE_CUDA_BACKEND

#include "../affineBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(8u);
InstantiateInvokeAffineBackSrcP2ForGeomType(8u);
InstantiateInvokeAffineBackSrcP3ForGeomType(8u);
InstantiateInvokeAffineBackSrcP4ForGeomType(8u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
