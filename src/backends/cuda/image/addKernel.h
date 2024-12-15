#include <backends/cuda/streamCtx.h>
#include <common/arithmetic/binary_operators.h>
#include <common/image/functors/srcSrcFunctor.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace opp::image::cuda
{

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddSrcSrc(const SrcT *aSrc1, size_t pitchSrc1, const SrcT *aSrc2, size_t pitchSrc2, DstT *aDst,
                     size_t pitchDst, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
