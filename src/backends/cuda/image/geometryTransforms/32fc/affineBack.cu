#if OPP_ENABLE_CUDA_BACKEND

#include "../affineBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlpha(32fc);
InstantiateInvokeAffineBackSrcP2ForGeomType(32fc);
InstantiateInvokeAffineBackSrcP3ForGeomType(32fc);
InstantiateInvokeAffineBackSrcP4ForGeomType(32fc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
