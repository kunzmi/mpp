#if OPP_ENABLE_CUDA_BACKEND

#include "../affineBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(32u);
InstantiateInvokeAffineBackSrcP2ForGeomType(32u);
InstantiateInvokeAffineBackSrcP3ForGeomType(32u);
InstantiateInvokeAffineBackSrcP4ForGeomType(32u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
