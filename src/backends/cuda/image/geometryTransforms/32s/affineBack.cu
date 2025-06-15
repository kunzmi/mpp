#if OPP_ENABLE_CUDA_BACKEND

#include "../affineBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(32s);
InstantiateInvokeAffineBackSrcP2ForGeomType(32s);
InstantiateInvokeAffineBackSrcP3ForGeomType(32s);
InstantiateInvokeAffineBackSrcP4ForGeomType(32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
