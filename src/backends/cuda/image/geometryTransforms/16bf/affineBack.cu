#if OPP_ENABLE_CUDA_BACKEND

#include "../affineBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16bf);
InstantiateInvokeAffineBackSrcP2ForGeomType(16bf);
InstantiateInvokeAffineBackSrcP3ForGeomType(16bf);
InstantiateInvokeAffineBackSrcP4ForGeomType(16bf);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
