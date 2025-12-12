#include "../affineBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(32fc);
InstantiateInvokeAffineBackSrcP2ForGeomType(32fc);
InstantiateInvokeAffineBackSrcP3ForGeomType(32fc);
InstantiateInvokeAffineBackSrcP4ForGeomType(32fc);

} // namespace mpp::image::cuda
