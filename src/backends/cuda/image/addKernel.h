#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/tupel.h>
#include <cuda_runtime.h>

#include <common/arithmetic/binary_operators.h>
#include <common/image/functors/srcSrcFunctor.h>
#include <common/image/pixelTypes.h>
#include <common/vectorTypes.h>

namespace opp::cuda::image
{
template <typename SrcT, typename ComputeT = opp::image::default_compute_type_for_t<SrcT>, typename DstT,
          int hardwareMajor = 0, int hardwareMinor = 0>
void InvokeAddSrcSrc(const SrcT *aSrc1, size_t pitchSrc1, const SrcT *aSrc2, size_t pitchSrc2, DstT *aDst,
                     size_t pitchDst, const opp::image::Size2D &aSize);

} // namespace opp::cuda::image
