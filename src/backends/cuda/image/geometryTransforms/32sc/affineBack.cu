#include "../affineBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(32sc);
InstantiateInvokeAffineBackSrcP2ForGeomType(32sc);
InstantiateInvokeAffineBackSrcP3ForGeomType(32sc);
InstantiateInvokeAffineBackSrcP4ForGeomType(32sc);

} // namespace mpp::image::cuda
