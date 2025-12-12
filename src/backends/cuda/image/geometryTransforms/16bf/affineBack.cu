#include "../affineBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16bf);
InstantiateInvokeAffineBackSrcP2ForGeomType(16bf);
InstantiateInvokeAffineBackSrcP3ForGeomType(16bf);
InstantiateInvokeAffineBackSrcP4ForGeomType(16bf);

} // namespace mpp::image::cuda
