#include "convertedArgument.h"
#include "convertedFunction.h"
#include <cstddef>
#include <string>
#include <utilities/nppParser/function.h>
#include <utilities/nppParser/nppParser.h>
#include <utility>

namespace opp::utilities::nppParser
{
ConvertedArgument::ConvertedArgument(const ConvertedFunction &aFunction, std::string aType, std::string aName)
    : mFunction(aFunction), mType(std::move(aType)), mName(std::move(aName))
{
}
ConvertedArgument::ConvertedArgument( // NOLINT(hicpp-function-size,readability-function-size)
    const Argument *aArgument, const ConvertedFunction &aFunction)
    : mArgument(aArgument), mFunction(aFunction), mType(aArgument->type), mName(aArgument->name),
      mCall(mArgument->name), mIsSkippedInCall(false)
{
    const size_t arrayPos = aArgument->type.find('[');
    if (arrayPos != std::string::npos)
    {
        // array input argument: remove the [] (or [][]) from type and add it to the name
        mType = aArgument->type.substr(0, arrayPos);
        mName += aArgument->type.substr(arrayPos);
    }
    {
        if (aArgument->name == "hpBufferSize" || aArgument->name == "nBufferSize" ||
            aArgument->name == "hpDeviceMemoryBufferSize" || aArgument->name == "oSizeROI" ||
            aArgument->name == "pSrcDst" /*inplace*/ || (aArgument->name == "pDst" && mFunction.IsInplace()) ||
            aArgument->name == "aSrc" ||
            (aArgument->name == "pSrc1" && mFunction.Name() != "AddProduct") /*first source input*/ ||
            aArgument->name == "oSrcSizeROI" || aArgument->name == "oTplRoiSize" || aArgument->name == "oSrcRoiSize" ||
            aArgument->name == "oSrcROI" || aArgument->name == "oSrcSize" || aArgument->name == "oDstSizeROI" ||
            aArgument->name == "oDstSize" || aArgument->name == "oSrcRectROI" || aArgument->name == "oDstRectROI" ||
            (aArgument->name == "oDstROI" && !mFunction.IsGetBufferSizeFunction() &&
             aArgument->type.find("NppPointPolar") == std::string::npos) /*keep the argument for FilterHoughLineRegion*/
            || aArgument->name == "oSrcOffset" || aArgument->name == "oROI")
        {
            mIsSkippedInDeclaration = true;
        }
        else if (aArgument->name == "pSrc") // only one source input
        {
            mIsSkippedInDeclaration = !mFunction.IsInplace();
        }
        else
        {
            mIsSkippedInDeclaration = IsStep();
        }
    }

    if (aArgument->type == "NppStreamContext")
    {
        mType = "const NppStreamContext &";
    }
    else if (aArgument->name == "pSrc1" || aArgument->name == "pSrc2" || aArgument->name == "pSrc" ||
             aArgument->name == "pTpl" || aArgument->name == "pSrcY" || aArgument->name == "pSrcCbCr")
    {
        const std::string type         = ConvertNppType(aArgument->type);
        const std::string channelCount = std::to_string(mFunction.GetInChannels());

        mType = std::string("const Image") + type + "C" + channelCount + "View &";
    }
    else if (aArgument->name == "pDst" && mFunction.IsOutputPlanar())
    {
        const std::string channelCount = std::to_string(mFunction.GetOutChannels());

        if (aArgument->type == "Npp8u *[" + channelCount + "]")
        {
            mType = "Image8uC1View *";
        }
        else if (aArgument->type == "Npp8s *[" + channelCount + "]")
        {
            mType = "Image8sC1View *";
        }
        else if (aArgument->type == "Npp16u *[" + channelCount + "]")
        {
            mType = "Image16uC1View *";
        }
        else if (aArgument->type == "Npp16s *[" + channelCount + "]")
        {
            mType = "Image16sC1View *";
        }
        else if (aArgument->type == "Npp16f *[" + channelCount + "]")
        {
            mType = "Image16fC1View *";
        }
        else if (aArgument->type == "Npp32u *[" + channelCount + "]")
        {
            mType = "Image32uC1View *";
        }
        else if (aArgument->type == "Npp32s *[" + channelCount + "]")
        {
            mType = "Image32sC1View *";
        }
        else if (aArgument->type == "Npp32f *[" + channelCount + "]")
        {
            mType = "Image32fC1View *";
        }
        else if (aArgument->type == "Npp64f *[" + channelCount + "]")
        {
            mType = "Image64fC1View *";
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if ((aArgument->name == "pDstY" || aArgument->name == "pDstCbCr") && mFunction.IsOutputPlanar())
    {
        if (aArgument->type == "Npp8u *")
        {
            mType = "Image8uC1View &";
        }
        else if (aArgument->type == "Npp8s *")
        {
            mType = "Image8sC1View &";
        }
        else if (aArgument->type == "Npp16u *")
        {
            mType = "Image16uC1View &";
        }
        else if (aArgument->type == "Npp16s *")
        {
            mType = "Image16sC1View &";
        }
        else if (aArgument->type == "Npp16f *")
        {
            mType = "Image16fC1View &";
        }
        else if (aArgument->type == "Npp32u *")
        {
            mType = "Image32uC1View &";
        }
        else if (aArgument->type == "Npp32s *")
        {
            mType = "Image32sC1View &";
        }
        else if (aArgument->type == "Npp32f *")
        {
            mType = "Image32fC1View &";
        }
        else if (aArgument->type == "Npp64f *")
        {
            mType = "Image64fC1View &";
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if (aArgument->name == "pDst" && !mFunction.IsOutputPlanar())
    {
        const int channelCount = mFunction.GetOutChannels();

        if (aFunction.InnerFunction().name == "nppiLUTPalette_8u24u_C1R_Ctx" ||
            aFunction.InnerFunction().name == "nppiLUTPalette_16u24u_C1R_Ctx")
        {
            mType = "Image8uC3View &";
        }
        else if (aFunction.InnerFunction().name == "nppiLUTPalette_8u32u_C1R_Ctx" ||
                 aFunction.InnerFunction().name == "nppiLUTPalette_16u32u_C1R_Ctx")
        {
            mType = "Image8uC4View &";
        }
        else if (aFunction.InnerFunction().name.find("QualityIndex") != std::string::npos)
        {
            mType = "opp::cuda::DevVarView<float> &";
        }
        else if (aArgument->type == "Npp8u *" || aArgument->type == "Npp8s *" || aArgument->type == "Npp16u *" ||
                 aArgument->type == "Npp16s *" || aArgument->type == "Npp16sc *" || aArgument->type == "Npp16f *" ||
                 aArgument->type == "Npp32u *" || aArgument->type == "Npp32s *" || aArgument->type == "Npp32sc *" ||
                 aArgument->type == "Npp32f *" || aArgument->type == "Npp32fc *" || aArgument->type == "Npp64f *")
        {
            mType = GetImageType(aArgument->type, channelCount);
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if (aArgument->name == "pSrcDst" && mIsSkippedInDeclaration)
    {
        // it is not really necessary to set this type, but it helps when setting the function return type:
        const int channelCount = mFunction.GetOutChannels();

        if (aArgument->type == "Npp8u *" || aArgument->type == "Npp8s *" || aArgument->type == "Npp16u *" ||
            aArgument->type == "Npp16s *" || aArgument->type == "Npp16sc *" || aArgument->type == "Npp16f *" ||
            aArgument->type == "Npp32u *" || aArgument->type == "Npp32s *" || aArgument->type == "Npp32sc *" ||
            aArgument->type == "Npp32f *" || aArgument->type == "Npp32fc *" || aArgument->type == "Npp64f *")
        {
            mType = GetImageType(aArgument->type, channelCount);
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if (aArgument->name == "pDstX" || aArgument->name == "pDstY" || aArgument->name == "pDstMag" ||
             aArgument->name == "pDstVoronoi" || aArgument->name == "pDstVoronoiIndices" ||
             aArgument->name == "pDstVoronoiRelativeManhattanDistances")
    {
        if (aArgument->type == "Npp16s *")
        {
            mType = "Image16sC1View &";
        }
        else if (aArgument->type == "Npp32s *")
        {
            mType = "Image32sC1View &";
        }
        else if (aArgument->type == "Npp32f *")
        {
            mType = "Image32fC1View &";
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if (aArgument->name == "pDstVoronoiAbsoluteManhattanDistances")
    {
        if (aArgument->type == "Npp16u *")
        {
            mType = "Image16uC1View &";
        }
        else if (aArgument->type == "Npp32s *")
        {
            mType = "Image32sC1View &";
        }
        else if (aArgument->type == "Npp32f *")
        {
            mType = "Image32fC1View &";
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if (aArgument->name == "pDstAngle")
    {
        if (aArgument->type == "Npp32f *")
        {
            mType = "Image32fC1View &";
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if (aArgument->name == "pDstTransform")
    {
        if (aArgument->type == "Npp16u *")
        {
            mType = "Image16uC1View &";
        }
        else if (aArgument->type == "Npp32f *")
        {
            mType = "Image32fC1View &";
        }
        else if (aArgument->type == "Npp64f *")
        {
            mType = "Image64fC1View &";
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if (aArgument->name == "pMarkerLabels")
    {
        if (aArgument->type == "Npp32u *")
        {
            mType = "Image32uC1View &";
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if (aArgument->name == "pSqr")
    {
        if (aArgument->type == "Npp32s *" || aArgument->type == "const Npp32s *")
        {
            mType = "Image32sC1View &";
        }
        else if (aArgument->type == "Npp64f *" || aArgument->type == "const Npp64f *")
        {
            mType = "Image64fC1View &";
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if (aArgument->name == "pMask")
    {
        if (mFunction.IsMasked())
        {
            mType = "const Image8uC1View &";
        }
        else if (aArgument->type == "const Npp8u *")
        {
            mType = "const opp::cuda::DevVarView<byte> &";
        }
        else if (aArgument->type == "const Npp32s *")
        {
            mType = "const opp::cuda::DevVarView<int> &";
        }
        else if (aArgument->type == "const Npp32f *")
        {
            mType = "const opp::cuda::DevVarView<float> &";
        }
        else
        {
            mType = "Unknown";
        }
    }
    if (aArgument->name == "pKernel")
    {
        if (aArgument->type == "const Npp32s *")
        {
            mType = "const opp::cuda::DevVarView<int> &";
        }
        else if (aArgument->type == "const Npp32f *")
        {
            mType = "const opp::cuda::DevVarView<float> &";
        }
        else if (aArgument->type == "const Npp64f *")
        {
            mType = "const opp::cuda::DevVarView<double> &";
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if (aArgument->name == "pLevels")
    {
        if (aArgument->type == "const Npp32s *")
        {
            mType = "const opp::cuda::DevVarView<int> &";
        }
        else if (aArgument->type == "const Npp32f *")
        {
            mType = "const opp::cuda::DevVarView<float> &";
        }
        else if (aArgument->type == "const Npp32s *[3]" || aArgument->type == "const Npp32s *[4]")
        {
            mType = "opp::cuda::DevVarView<int>";
            mCall = "pLevelsPtrList";
        }
        else if (aArgument->type == "Npp8u *[3]") // LUT_Trilinear
        {
            /*mType = "opp::cuda::DevVarView<byte>";
            mCall = "pLevelsPtrList";*/
        }
        else if (aArgument->type == "const Npp32f *[3]" || aArgument->type == "const Npp32f *[4]")
        {
            mType = "opp::cuda::DevVarView<float>";
            mCall = "pLevelsPtrList";
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if (aArgument->name == "pTables")
    {
        if (aArgument->type == "const Npp8u *[3]" || aArgument->type == "const Npp8u *[4]")
        {
            mType = "opp::cuda::DevVarView<byte>";
            mCall = "pTablesPtrList";
        }
        else if (aArgument->type == "const Npp16u *[3]" || aArgument->type == "const Npp16u *[4]")
        {
            mType = "opp::cuda::DevVarView<ushort>";
            mCall = "pTablesPtrList";
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if (aArgument->name == "pValues")
    {
        if (aArgument->type == "const Npp32s *")
        {
            mType = "const opp::cuda::DevVarView<int> &";
        }
        else if (aArgument->type == "const Npp32f *")
        {
            mType = "const opp::cuda::DevVarView<float> &";
        }
        else if (aArgument->type == "const Npp32s *[3]" || aArgument->type == "const Npp32s *[4]")
        {
            mType = "opp::cuda::DevVarView<int>";
            mCall = "pValuesPtrList";
        }
        else if (aArgument->type == "Npp32u *") // LUT_Trilinear
        {
            mType = "opp::cuda::DevVarView<Pixel8uC4>";
        }
        else if (aArgument->type == "const Npp32f *[3]" || aArgument->type == "const Npp32f *[4]")
        {
            mType = "opp::cuda::DevVarView<float>";
            mCall = "pValuesPtrList";
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if (aArgument->name == "pTable")
    {
        const std::string channel = NPPParser::GetTypeString(mFunction.InnerFunction().name);
        if (channel == "8u24u" || channel == "16u24u")
        {
            mType = "const opp::cuda::DevVarView<Pixel8uC3> &";
        }
        else if (channel == "8u32u" || channel == "16u32u")
        {
            mType = "const opp::cuda::DevVarView<Pixel8uC4> &";
        }
        else if (aArgument->type == "const Npp8u *")
        {
            mType = "const opp::cuda::DevVarView<Pixel8uC1> &";
        }
        else if (aArgument->type == "const Npp16u *")
        {
            mType = "const opp::cuda::DevVarView<Pixel16uC1> &";
        }
        else if (aArgument->type == "const Npp32u *")
        {
            mType = "const opp::cuda::DevVarView<Pixel32uC1> &";
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if (aArgument->name == "pSum" || aArgument->name == "pMean" || aArgument->name == "pStdDev" ||
             aArgument->name == "pNorm" || aArgument->name == "pNormDiff" || aArgument->name == "pNormRel" ||
             aArgument->name == "pError")
    {
        if (aArgument->type == "Npp64f *")
        {
            mType = "opp::cuda::DevVarView<double> &";
        }
        else if (aArgument->type == "Npp64s *")
        {
            mType = "opp::cuda::DevVarView<long64> &";
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if (aArgument->name == "aSum" || aArgument->name == "aMean" || aArgument->name == "aNorm" ||
             aArgument->name == "aNormDiff" || aArgument->name == "aNormRel")
    {
        if (aArgument->type == "Npp64f[3]" || aArgument->type == "Npp64f[4]")
        {
            mType = "opp::cuda::DevVarView<double> &";
            mName = mName.substr(0, mName.size() - 3);
        }
        else if (aArgument->type == "Npp64s[3]" || aArgument->type == "Npp64s[4]")
        {
            mType = "opp::cuda::DevVarView<long64> &";
            mName = mName.substr(0, mName.size() - 3);
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if (aArgument->name == "pMSE" || aArgument->name == "pPSNR" || aArgument->name == "pSSIM" ||
             aArgument->name == "pMSSSIM")
    {
        if (aArgument->type == "Npp32f *")
        {
            mType = "opp::cuda::DevVarView<float> &";
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if ((aArgument->name == "pMin" || aArgument->name == "pMinValue" || aArgument->name == "pMax" ||
              aArgument->name == "pMaxValue") &&
             mFunction.InnerFunction().category == "statistics")
    {
        if (aArgument->type == "Npp8u *")
        {
            mType = "opp::cuda::DevVarView<byte> &";
        }
        else if (aArgument->type == "Npp8s *")
        {
            mType = "opp::cuda::DevVarView<sbyte> &";
        }
        else if (aArgument->type == "Npp16u *")
        {
            mType = "opp::cuda::DevVarView<ushort> &";
        }
        else if (aArgument->type == "Npp16s *")
        {
            mType = "opp::cuda::DevVarView<short> &";
        }
        else if (aArgument->type == "Npp32u *")
        {
            mType = "opp::cuda::DevVarView<uint> &";
        }
        else if (aArgument->type == "Npp32s *")
        {
            mType = "opp::cuda::DevVarView<int> &";
        }
        else if (aArgument->type == "Npp32f *")
        {
            mType = "opp::cuda::DevVarView<float> &";
        }
        else if (aArgument->type == "Npp64f *")
        {
            mType = "opp::cuda::DevVarView<double> &";
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if ((aArgument->name == "aMin" || aArgument->name == "aMinValue" || aArgument->name == "aMax" ||
              aArgument->name == "aMaxValue") &&
             mFunction.InnerFunction().category == "statistics")
    {
        if (aArgument->type == "Npp8u[3]" || aArgument->type == "Npp8u[4]")
        {
            mType = "opp::cuda::DevVarView<byte> &";
            mName = mName.substr(0, mName.size() - 3);
        }
        else if (aArgument->type == "Npp8s[3]" || aArgument->type == "Npp8s[4]")
        {
            mType = "opp::cuda::DevVarView<sbyte> &";
            mName = mName.substr(0, mName.size() - 3);
        }
        else if (aArgument->type == "Npp16u[3]" || aArgument->type == "Npp16u[4]")
        {
            mType = "opp::cuda::DevVarView<ushort> &";
            mName = mName.substr(0, mName.size() - 3);
        }
        else if (aArgument->type == "Npp16s[3]" || aArgument->type == "Npp16s[4]")
        {
            mType = "opp::cuda::DevVarView<short> &";
            mName = mName.substr(0, mName.size() - 3);
        }
        else if (aArgument->type == "Npp32u[3]" || aArgument->type == "Npp32u[4]")
        {
            mType = "opp::cuda::DevVarView<uint> &";
            mName = mName.substr(0, mName.size() - 3);
        }
        else if (aArgument->type == "Npp32s[3]" || aArgument->type == "Npp32s[4]")
        {
            mType = "opp::cuda::DevVarView<int> &";
            mName = mName.substr(0, mName.size() - 3);
        }
        else if (aArgument->type == "Npp32f[3]" || aArgument->type == "Npp32f[4]")
        {
            mType = "opp::cuda::DevVarView<float> &";
            mName = mName.substr(0, mName.size() - 3);
        }
        else if (aArgument->type == "Npp64f[3]" || aArgument->type == "Npp64f[4]")
        {
            mType = "opp::cuda::DevVarView<double> &";
            mName = mName.substr(0, mName.size() - 3);
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if (aArgument->name == "pIndexX" || aArgument->name == "pIndexY" || aArgument->name == "pCounts" ||
             aArgument->name == "aIndexX" || aArgument->name == "aIndexY" || aArgument->name == "aCounts")
    {
        if (aArgument->type == "int *" || aArgument->type == "int[3]" || aArgument->type == "int[4]")
        {
            mType = "opp::cuda::DevVarView<int> &";
            if (aArgument->type.find('[') != std::string::npos)
            {
                mName = mName.substr(0, mName.size() - 3);
            }
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if (aArgument->name == "pHist")
    {
        if (aArgument->type == "Npp32s *")
        {
            mType = "opp::cuda::DevVarView<int> &";
        }
        else if (aArgument->type == "Npp32s *[3]" || aArgument->type == "Npp32s *[4]")
        {
            mType = "opp::cuda::DevVarView<int>";
            mCall = "pHistPtrList";
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if (aArgument->name == "pMinIndex" || aArgument->name == "pMaxIndex")
    {
        if (aArgument->type == "NppiPoint *")
        {
            mType = "opp::cuda::DevVarView<NppiPoint> &";
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if (aArgument->name == "pXMap" || aArgument->name == "pYMap")
    {
        if (aArgument->type == "const Npp32f *")
        {
            mType = "const Image32fC1View &";
        }
        else if (aArgument->type == "const Npp64f *")
        {
            mType = "const Image64fC1View &";
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if (aArgument->name == "nConstant" && aArgument->type == "const Npp32u")
    {
        mType = "Npp32u";
    }
    else if ((aArgument->name == "aDp" || aArgument->name == "pDp") && mFunction.Name() == "DotProd")
    {
        if (aArgument->type == "Npp64f *")
        {
            mType = "opp::cuda::DevVarView<double> &";
        }
        else if (aArgument->type == "Npp64f[2]" || aArgument->type == "Npp64f[3]" || aArgument->type == "Npp64f[4]")
        {
            mType = "opp::cuda::DevVarView<double> &";
            mName = mName.substr(0, mName.find('['));
        }
        else
        {
            mType = "Unknown";
        }
    }
    else if (aArgument->type == "Npp8u" || aArgument->type == "const Npp8u")
    {
        mType = "const Pixel8uC1 &";
    }
    else if (aArgument->type == "const Npp8u[2]")
    {
        mType = "const Pixel8uC2 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp8u[2]")
    {
        mType = "Pixel8uC2 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "const Npp8u[3]")
    {
        mType = "const Pixel8uC3 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp8u[3]")
    {
        mType = "Pixel8uC3 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "const Npp8u[4]")
    {
        mType = "const Pixel8uC4 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp8u[4]")
    {
        mType = "Pixel8uC4 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp8s" || aArgument->type == "const Npp8s")
    {
        mType = "const Pixel8sC1 &";
    }
    else if (aArgument->type == "const Npp8s[2]")
    {
        mType = "const Pixel8sC2 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp8s[2]")
    {
        mType = "Pixel8sC2 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "const Npp8s[3]")
    {
        mType = "const Pixel8sC3 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp8s[3]")
    {
        mType = "Pixel8sC3 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "const Npp8s[4]")
    {
        mType = "const Pixel8sC4 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp8s[4]")
    {
        mType = "Pixel8sC4 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp16u" || aArgument->type == "const Npp16u")
    {
        mType = "const Pixel16uC1 &";
    }
    else if (aArgument->type == "const Npp16u[2]")
    {
        mType = "const Pixel16uC2 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp16u[2]")
    {
        mType = "Pixel16uC2 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "const Npp16u[3]")
    {
        mType = "const Pixel16uC3 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp16u[3]")
    {
        mType = "Pixel16uC3 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "const Npp16u[4]")
    {
        mType = "const Pixel16uC4 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp16u[4]")
    {
        mType = "Pixel16uC4 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp16s" || aArgument->type == "const Npp16s")
    {
        mType = "const Pixel16sC1 &";
    }
    else if (aArgument->type == "const Npp16s[2]")
    {
        mType = "const Pixel16sC2 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp16s[2]")
    {
        mType = "Pixel16sC2 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "const Npp16s[3]")
    {
        mType = "const Pixel16sC3 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp16s[3]")
    {
        mType = "Pixel16sC3 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "const Npp16s[4]")
    {
        mType = "const Pixel16sC4 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp16s[4]")
    {
        mType = "Pixel16sC4 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp16sc" || aArgument->type == "const Npp16sc")
    {
        mType = "const Pixel16scC1 &";
    }
    else if (aArgument->type == "const Npp16sc[2]")
    {
        mType = "const Pixel16scC2 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp16sc[2]")
    {
        mType = "Pixel16scC2 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "const Npp16sc[3]")
    {
        mType = "const Pixel16scC3 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp16sc[3]")
    {
        mType = "Pixel16scC3 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "const Npp16sc[4]")
    {
        mType = "const Pixel16scC4 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp16sc[4]")
    {
        mType = "Pixel16scC4 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp32u" || aArgument->type == "const Npp32u")
    {
        mType = "const Pixel32uC1 &";
    }
    else if (aArgument->type == "const Npp32u[2]")
    {
        mType = "const Pixel32uC2 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp32u[2]")
    {
        mType = "Pixel32uC2 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "const Npp32u[3]")
    {
        mType = "const Pixel32uC3 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp32u[3]")
    {
        mType = "Pixel32uC3 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "const Npp32u[4]")
    {
        mType = "const Pixel32uC4 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp32u[4]")
    {
        mType = "Pixel32uC4 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if ((aArgument->type == "Npp32s" && !IsStep() && aArgument->name != "nAnchor" &&
              aArgument->name != "nMaskSize") ||
             aArgument->type == "const Npp32s")
    {
        mType = "const Pixel32sC1 &";
    }
    else if (aArgument->type == "const Npp32s[2]")
    {
        mType = "const Pixel32sC2 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp32s[2]")
    {
        mType = "Pixel32sC2 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "const Npp32s[3]")
    {
        mType = "const Pixel32sC3 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp32s[3]")
    {
        mType = "Pixel32sC3 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "const Npp32s[4]")
    {
        mType = "const Pixel32sC4 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp32s[4]")
    {
        mType = "Pixel32sC4 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp32sc" || aArgument->type == "const Npp32sc")
    {
        mType = "const Pixel32scC1 &";
    }
    else if (aArgument->type == "const Npp32sc[2]")
    {
        mType = "const Pixel32scC2 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp32sc[2]")
    {
        mType = "Pixel32scC2 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "const Npp32sc[3]")
    {
        mType = "const Pixel32scC3 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp32sc[3]")
    {
        mType = "Pixel32scC3 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "const Npp32sc[4]")
    {
        mType = "const Pixel32scC4 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp32sc[4]")
    {
        mType = "Pixel32scC4 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp32f" || aArgument->type == "const Npp32f")
    {
        mType = "const Pixel32fC1 &";
    }
    else if (aArgument->type == "const Npp32f[2]")
    {
        mType = "const Pixel32fC2 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp32f[2]")
    {
        mType = "Pixel32fC2 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "const Npp32f[3]")
    {
        mType = "const Pixel32fC3 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp32f[3]")
    {
        mType = "Pixel32fC3 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "const Npp32f[4]")
    {
        mType = "const Pixel32fC4 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp32f[4]")
    {
        mType = "Pixel32fC4 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp32fc" || aArgument->type == "const Npp32fc")
    {
        mType = "const Pixel32fcC1 &";
    }
    else if (aArgument->type == "const Npp32fc[2]")
    {
        mType = "const Pixel32fcC2 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp32fc[2]")
    {
        mType = "Pixel32fcC2 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "const Npp32fc[3]")
    {
        mType = "const Pixel32fcC3 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp32fc[3]")
    {
        mType = "Pixel32fcC3 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "const Npp32fc[4]")
    {
        mType = "const Pixel32fcC4 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp32fc[4]")
    {
        mType = "Pixel32fcC4 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp64f" || aArgument->type == "const Npp64f")
    {
        mType = "const Pixel64fC1 &";
    }
    else if (aArgument->type == "const Npp64f[2]")
    {
        mType = "const Pixel64fC2 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp64f[2]")
    {
        mType = "Pixel64fC2 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "const Npp64f[3]")
    {
        mType = "const Pixel64fC3 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp64f[3]")
    {
        mType = "Pixel64fC3 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "const Npp64f[4]")
    {
        mType = "const Pixel64fC4 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->type == "Npp64f[4]")
    {
        mType = "Pixel64fC4 &";
        mName = mName.substr(0, mName.find('['));
    }
    else if (aArgument->name == "pConstant" && aArgument->type.find('*') != std::string::npos)
    {
        if (aArgument->type.find("const") == std::string::npos)
        {
            mType = "opp::cuda::DevVarView<Pixel8uC1> &";
        }
        else
        {
            mType = "const opp::cuda::DevVarView<Pixel8uC1> &";
        }
    }
    else if (aArgument->name == "pConstants" && aArgument->type.find('*') != std::string::npos)
    {
        if (mFunction.Name().find("Compare") != std::string::npos)
        {
            int channels = mFunction.GetInChannels();
            if (mFunction.IsAlphaIgnored())
            {
                channels = 3;
            }
            mType = "const Pixel" + mFunction.GetTypeString() + "C" + std::to_string(channels) + " &";
        }
        else
        {
            std::string conststr;
            if (aArgument->type.find("const") != std::string::npos)
            {
                conststr = "const ";
            }
            int channels = mFunction.GetInChannels();
            if (mFunction.IsAlphaIgnored())
            {
                channels = 3;
            }
            mType = conststr + "opp::cuda::DevVarView<Pixel" + mFunction.GetTypeString() + "C" +
                    std::to_string(channels) + "> &";
        }
    }
    else if (aArgument->name.find("Buffer") != std::string::npos && aArgument->type == "Npp8u *")
    {
        mType = "opp::cuda::DevVarView<byte> &";
    }
    else if (aArgument->name == "oDstROI" && mFunction.Name() == "ResizeAdvancedGetBufferHostSize")
    {
        mType = "const " + mFunction.GetImageViewType() + " &";
    }
    else if (aArgument->name == "aCoeffs" && mFunction.Name().substr(0, 10) == "WarpAffine")
    {
        mType = "const AffineTransformation<double> &";
        mName = "aCoeffs"; // remove the "[3][2]"
        mCall = "reinterpret_cast<const double(*)[3]>(&aCoeffs)";
    }
    else if (aArgument->name == "aSrcQuad" &&
             (mFunction.Name().substr(0, 10) == "WarpAffine" || mFunction.Name().substr(0, 10) == "WarpPerspe"))
    {
        mType = "const Quad<double> &";
        mName = "aSrcQuad"; // remove the "[4][2]"
        mCall = "reinterpret_cast<const double(*)[2]>(&aSrcQuad)";
    }
    else if (aArgument->name == "aDstQuad" &&
             (mFunction.Name().substr(0, 10) == "WarpAffine" || mFunction.Name().substr(0, 10) == "WarpPerspe"))
    {
        mType = "const Quad<double> &";
        mName = "aDstQuad"; // remove the "[4][2]"
        mCall = "reinterpret_cast<const double(*)[2]>(&aDstQuad)";
    }
    else if (aArgument->name == "aCoeffs" && mFunction.Name().substr(0, 10) == "WarpPerspe")
    {
        mType = "const PerspectiveTransformation<double> &";
        mName = "aCoeffs"; // remove the "[3][3]"
        mCall = "reinterpret_cast<const double(*)[3]>(&aCoeffs)";
    }
}
ConvertedArgument::ConvertedArgument(const Argument *aArgument, ConvertedArgument *aLinkedArgument,
                                     const ConvertedFunction &aFunction)
    : ConvertedArgument(aArgument, aFunction)
{
    mLinkedArgument = aLinkedArgument;
}
std::string ConvertedArgument::ToStringDeclaration() const
{
    if (mIsSkippedInDeclaration)
    {
        return {};
    }

    if (mType.size() > 2 && (mType[mType.size() - 1] == '&' || mType[mType.size() - 1] == '*'))
    {
        return mType + mName;
    }
    return mType + " " + mName;
}
std::string ConvertedArgument::ToStringNppCall() const
{
    if (mIsSkippedInCall)
    {
        return {};
    }
    return mCall;
}

bool ConvertedArgument::IsInputImage() const
{
    if (mArgument == nullptr)
    {
        return false;
    }
    return mArgument->name == "aSrc" || mArgument->name == "pSrc" || mArgument->name == "pSrc1" ||
           mArgument->name == "pSrc2" || mArgument->name == "pTpl" || mArgument->name == "pXMap" ||
           mArgument->name == "pYMap" || mArgument->name == "pSrcY" || mArgument->name == "pSrcCbCr" ||
           mArgument->name == "pCompressedMarkerLabels";
}

bool ConvertedArgument::IsOutputImage() const
{
    return mName == "pSrcDst" || mName == "pSqr" ||
           (mName == "pDst" && mFunction.Name() != "QualityIndex" && mFunction.Name() != "QualityIndexA") ||
           mName == "pMarkerLabels" || mName == "pDstX" || mName == "pDstY" || mName == "pDstCbCr" ||
           mName == "pDstMag" || mName == "pDstAngle" || mName == "pDstVoronoi" || mName == "pDstVoronoiIndices" ||
           mName == "pDstVoronoiRelativeManhattanDistances" || mName == "pDstVoronoiAbsoluteManhattanDistances" ||
           mName == "pDstTransform" || mName == "pDst[2]" || mName == "pDst[3]" || mName == "pDst[4]" ||
           mName == "aDst[2]" || mName == "aDst[3]" || mName == "aDst[4]" || mName == "aDst[4]";
}

bool ConvertedArgument::IsMask() const
{
    if (mArgument != nullptr && mArgument->name == "pMask")
    {
        // there are also other mask parameters but which are not the image mask
        if (NPPParser::IsMasked(mFunction.InnerFunction()))
        {
            return true;
        }
    }
    return false;
}

bool ConvertedArgument::IsStep() const
{
    return mArgument->name.size() > 4 && mArgument->name.substr(mArgument->name.size() - 4) == "Step";
}

std::string ConvertedArgument::ConvertNppType(const std::string &aNPPType)
{
    size_t posNpp = aNPPType.find("Npp");
    if (posNpp == std::string::npos)
    {
        return aNPPType;
    }
    std::string newType = aNPPType.substr(posNpp + 3);

    posNpp = newType.find(' ');
    if (posNpp == std::string::npos)
    {
        return newType;
    }
    return newType.substr(0, posNpp);
}

std::string ConvertedArgument::GetImageType(const std::string &aNPPType, int aChannelCount)
{
    const std::string channelCount = std::to_string(aChannelCount);

    if (aNPPType == "Npp8u *")
    {
        return "Image8uC" + channelCount + "View &";
    }
    if (aNPPType == "Npp8s *")
    {
        return "Image8sC" + channelCount + "View &";
    }
    if (aNPPType == "Npp16u *")
    {
        return "Image16uC" + channelCount + "View &";
    }
    if (aNPPType == "Npp16s *")
    {
        return "Image16sC" + channelCount + "View &";
    }
    if (aNPPType == "Npp16sc *")
    {
        return "Image16scC" + channelCount + "View &";
    }
    if (aNPPType == "Npp16f *")
    {
        return "Image16fC" + channelCount + "View &";
    }
    if (aNPPType == "Npp32u *")
    {
        return "Image32uC" + channelCount + "View &";
    }
    if (aNPPType == "Npp32s *")
    {
        return "Image32sC" + channelCount + "View &";
    }
    if (aNPPType == "Npp32sc *")
    {
        return "Image32scC" + channelCount + "View &";
    }
    if (aNPPType == "Npp32f *")
    {
        return "Image32fC" + channelCount + "View &";
    }
    if (aNPPType == "Npp32fc *")
    {
        return "Image32fcC" + channelCount + "View &";
    }
    if (aNPPType == "Npp64f *")
    {
        return "Image64fC" + channelCount + "View &";
    }
    return "Unknown";
}
} // namespace opp::utilities::nppParser
