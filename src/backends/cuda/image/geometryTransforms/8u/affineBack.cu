#include "../affineBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(8u);
InstantiateInvokeAffineBackSrcP2ForGeomType(8u);
InstantiateInvokeAffineBackSrcP3ForGeomType(8u);
InstantiateInvokeAffineBackSrcP4ForGeomType(8u);

} // namespace mpp::image::cuda
