#pragma once

#include "image8uC1View.h"
#include <backends/cuda/devVarView.h>
#include <backends/npp/nppException.h>
#include <common/defines.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/image/sizePitched.h>
#include <common/safeCast.h>
#include <nppdefs.h>

namespace opp::image::npp
{

class Image8uC1 : public Image8uC1View
{

  public:
    Image8uC1() = delete;
    Image8uC1(int aWidth, int aHeight);
    explicit Image8uC1(const Size2D &aSize);

    ~Image8uC1();

    Image8uC1(const Image8uC1 &) = delete;
    Image8uC1(Image8uC1 &&aOther) noexcept;

    Image8uC1 &operator=(const Image8uC1 &) = delete;
    Image8uC1 &operator=(Image8uC1 &&aOther) noexcept;

    void Set(const Npp8u nValue, const NppStreamContext &nppStreamCtx);
    void Set(const Pixel8uC1 &nValue, const Image8uC1 &pMask, const NppStreamContext &nppStreamCtx);
    void Copy(const Image8uC1 &pSrc, const NppStreamContext &nppStreamCtx);
    void Copy(const Image8uC1 &pSrc, const Image8uC1 &pMask, const NppStreamContext &nppStreamCtx);
    void Copy(const Image8uC1 &pSrc, const NppStreamContext &nppStreamCtx);
    void Copy(const Image8uC1 &pSrc, const NppStreamContext &nppStreamCtx);
    void CopyConstBorder(const Image8uC1 &pSrc, int nTopBorderHeight, int nLeftBorderWidth, const Pixel8uC1 &nValue,
                         const NppStreamContext &nppStreamCtx);
    void CopyReplicateBorder(const Image8uC1 &pSrc, int nTopBorderHeight, int nLeftBorderWidth,
                             const NppStreamContext &nppStreamCtx);
    void CopyWrapBorder(const Image8uC1 &pSrc, int nTopBorderHeight, int nLeftBorderWidth,
                        const NppStreamContext &nppStreamCtx);
    void CopySubpix(const Image8uC1 &pSrc, Npp32f nDx, Npp32f nDy, const NppStreamContext &nppStreamCtx);
    void Transpose(const Image8uC1 &pSrc, NppiSize oSrcROI, const NppStreamContext &nppStreamCtx);
    void Add(const Image8uC1 &pSrc1, const Pixel8uC1 &nConstant, int nScaleFactor,
             const NppStreamContext &nppStreamCtx);
    void Add(const Image8uC1 &pSrc1, const cuda::DevVarView<Pixel8uC1> &pConstant, int nScaleFactor,
             const NppStreamContext &nppStreamCtx);
    void Add(const Pixel8uC1 &nConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void Add(const cuda::DevVarView<Pixel8uC1> &pConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void Mul(const Image8uC1 &pSrc1, const Pixel8uC1 &nConstant, int nScaleFactor,
             const NppStreamContext &nppStreamCtx);
    void Mul(const Image8uC1 &pSrc1, const cuda::DevVarView<Pixel8uC1> &pConstant, int nScaleFactor,
             const NppStreamContext &nppStreamCtx);
    void Mul(const Pixel8uC1 &nConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void Mul(const cuda::DevVarView<Pixel8uC1> &pConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void MulScale(const Image8uC1 &pSrc1, const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx);
    void MulScale(const Image8uC1 &pSrc1, const cuda::DevVarView<Pixel8uC1> &pConstant,
                  const NppStreamContext &nppStreamCtx);
    void MulScale(const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx);
    void MulScale(const cuda::DevVarView<Pixel8uC1> &pConstant, const NppStreamContext &nppStreamCtx);
    void Sub(const Image8uC1 &pSrc1, const Pixel8uC1 &nConstant, int nScaleFactor,
             const NppStreamContext &nppStreamCtx);
    void Sub(const Image8uC1 &pSrc1, const cuda::DevVarView<Pixel8uC1> &pConstant, int nScaleFactor,
             const NppStreamContext &nppStreamCtx);
    void Sub(const Pixel8uC1 &nConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void Sub(const cuda::DevVarView<Pixel8uC1> &pConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void Div(const Image8uC1 &pSrc1, const Pixel8uC1 &nConstant, int nScaleFactor,
             const NppStreamContext &nppStreamCtx);
    void Div(const Image8uC1 &pSrc1, const cuda::DevVarView<Pixel8uC1> &pConstant, int nScaleFactor,
             const NppStreamContext &nppStreamCtx);
    void Div(const Pixel8uC1 &nConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void Div(const cuda::DevVarView<Pixel8uC1> &pConstant, int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void AbsDiff(const Image8uC1 &pSrc1, const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx);
    void AbsDiff(const Image8uC1 &pSrc1, cuda::DevVarView<Pixel8uC1> &pConstant, const NppStreamContext &nppStreamCtx);
    void Add(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void Add(const Image8uC1 &pSrc, int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void Mul(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void Mul(const Image8uC1 &pSrc, int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void MulScale(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, const NppStreamContext &nppStreamCtx);
    void MulScale(const Image8uC1 &pSrc, const NppStreamContext &nppStreamCtx);
    void Sub(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void Sub(const Image8uC1 &pSrc, int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void Div(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void Div(const Image8uC1 &pSrc, int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void Div_Round(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, NppRoundMode rndMode, int nScaleFactor,
                   const NppStreamContext &nppStreamCtx);
    void Div_Round(const Image8uC1 &pSrc, NppRoundMode rndMode, int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void AbsDiff(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, const NppStreamContext &nppStreamCtx);
    void Sqr(const Image8uC1 &pSrc, int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void Sqr(int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void Sqrt(const Image8uC1 &pSrc, int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void Sqrt(int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void Ln(const Image8uC1 &pSrc, int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void Ln(int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void Exp(const Image8uC1 &pSrc, int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void Exp(int nScaleFactor, const NppStreamContext &nppStreamCtx);
    void And(const Image8uC1 &pSrc1, const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx);
    void And(const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx);
    void Or(const Image8uC1 &pSrc1, const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx);
    void Or(const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx);
    void Xor(const Image8uC1 &pSrc1, const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx);
    void Xor(const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx);
    void RShift(const Image8uC1 &pSrc1, const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx);
    void RShift(const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx);
    void LShift(const Image8uC1 &pSrc1, const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx);
    void LShift(const Pixel8uC1 &nConstant, const NppStreamContext &nppStreamCtx);
    void And(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, const NppStreamContext &nppStreamCtx);
    void And(const Image8uC1 &pSrc, const NppStreamContext &nppStreamCtx);
    void Or(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, const NppStreamContext &nppStreamCtx);
    void Or(const Image8uC1 &pSrc, const NppStreamContext &nppStreamCtx);
    void Xor(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, const NppStreamContext &nppStreamCtx);
    void Xor(const Image8uC1 &pSrc, const NppStreamContext &nppStreamCtx);
    void Not(const Image8uC1 &pSrc, const NppStreamContext &nppStreamCtx);
    void Not(const NppStreamContext &nppStreamCtx);
    void AlphaComp(const Image8uC1 &pSrc1, const Pixel8uC1 &nAlpha1, const Image8uC1 &pSrc2, const Pixel8uC1 &nAlpha2,
                   NppiAlphaOp eAlphaOp, const NppStreamContext &nppStreamCtx);
    void AlphaPremul(const Image8uC1 &pSrc1, const Pixel8uC1 &nAlpha1, const NppStreamContext &nppStreamCtx);
    void AlphaPremul(const Pixel8uC1 &nAlpha1, const NppStreamContext &nppStreamCtx);
    void AlphaComp(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, NppiAlphaOp eAlphaOp,
                   const NppStreamContext &nppStreamCtx);
    void RGBToGray(const Image8uC1 &pSrc, const NppStreamContext &nppStreamCtx);
    void RGBToGray(const Image8uC1 &pSrc, const NppStreamContext &nppStreamCtx);
    void ColorToGray(const Image8uC1 &pSrc, const Npp32f aCoeffs[3], const NppStreamContext &nppStreamCtx);
    void ColorToGray(const Image8uC1 &pSrc, const Npp32f aCoeffs[3], const NppStreamContext &nppStreamCtx);
    void ColorToGray(const Image8uC1 &pSrc, const Npp32f aCoeffs[4], const NppStreamContext &nppStreamCtx);
    void GradientColorToGray(const Image8uC1 &pSrc, NppiNorm eNorm, const NppStreamContext &nppStreamCtx);
    void CompColorKey(const Image8uC1 &pSrc1, const Image8uC1 &pSrc2, const Pixel8uC1 &nColorKeyConst,
                      const NppStreamContext &nppStreamCtx);
    void ColorTwist32f(const Image8uC1 &pSrc, const Npp32f aTwist[3][4], const NppStreamContext &nppStreamCtx);
    void ColorTwist32f(const Npp32f aTwist[3][4], const NppStreamContext &nppStreamCtx);
    void ColorTwistBatch32f(Npp32f nMin, Npp32f nMax, NppiColorTwistBatchCXR *pBatchList, int nBatchSize,
                            const NppStreamContext &nppStreamCtx);
    void ColorTwistBatch32f(Npp32f nMin, Npp32f nMax, NppiColorTwistBatchCXR *pBatchList, int nBatchSize,
                            const NppStreamContext &nppStreamCtx);
    void LUT(const Image8uC1 &pSrc, const Npp32s *pValues, const Npp32s *pLevels, int nLevels,
             const NppStreamContext &nppStreamCtx);
    void LUT(const Npp32s *pValues, const Npp32s *pLevels, int nLevels, const NppStreamContext &nppStreamCtx);
    void LUT_Linear(const Image8uC1 &pSrc, const Npp32s *pValues, const Npp32s *pLevels, int nLevels,
                    const NppStreamContext &nppStreamCtx);
    void LUT_Linear(const Npp32s *pValues, const Npp32s *pLevels, int nLevels, const NppStreamContext &nppStreamCtx);
    void LUT_Cubic(const Image8uC1 &pSrc, const Npp32s *pValues, const Npp32s *pLevels, int nLevels,
                   const NppStreamContext &nppStreamCtx);
    void LUT_Cubic(const Npp32s *pValues, const Npp32s *pLevels, int nLevels, const NppStreamContext &nppStreamCtx);
    void LUTPalette(const Image8uC1 &pSrc, const Npp8u *pTable, int nBitSize, const NppStreamContext &nppStreamCtx);
};
} // namespace opp::image::npp
