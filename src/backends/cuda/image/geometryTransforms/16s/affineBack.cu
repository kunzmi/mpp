#if OPP_ENABLE_CUDA_BACKEND

#include "../affineBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16s);
InstantiateInvokeAffineBackSrcP2ForGeomType(16s);
InstantiateInvokeAffineBackSrcP3ForGeomType(16s);
InstantiateInvokeAffineBackSrcP4ForGeomType(16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
