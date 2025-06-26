#include "convertedFunction.h"
#include "convertedArgument.h"
#include <algorithm>
#include <common/safeCast.h>
#include <cstddef>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <utilities/nppParser/function.h>
#include <utilities/nppParser/nppParser.h>
#include <vector>

namespace mpp::utilities::nppParser
{
ConvertedFunction::ConvertedFunction(const Function &aFunction)
    : mFunction(aFunction), mReturnType(aFunction.returnType), mName(NPPParser::GetShortName(aFunction)),
      mIsStatic(false), mIsConst(false)
{
    if (mReturnType == "NppStatus")
    {
        mReturnType = "void";
    }

    if (IsCopyExtract() || IsCopyChannel())
    {
        // Add additional channel argument:
        mArguments.emplace_back(*this, "size_t", "aSrcChannel");

        const std::string channelString = NPPParser::GetChannelString(aFunction.name);

        if (channelString == "C2C1R")
        {
            std::stringstream ss;
            ss << "    if (aSrcChannel >= 2)" << std::endl;
            ss << "    {" << std::endl;
            ss << R"(        throw INVALIDARGUMENT(aSrcChannel, "Value must be in range [0..1] but provided value is: " << aSrcChannel);)"
               << std::endl;
            ss << "    }" << std::endl;
            mCallHeader = ss.str();
        }
        if (channelString == "C3C1R" || channelString == "C3CR")
        {
            std::stringstream ss;
            ss << "    if (aSrcChannel >= 3)" << std::endl;
            ss << "    {" << std::endl;
            ss << R"(        throw INVALIDARGUMENT(aSrcChannel, "Value must be in range [0..2] but provided value is: " << aSrcChannel);)"
               << std::endl;
            ss << "    }" << std::endl;
            mCallHeader = ss.str();
        }
        if (channelString == "C4C1R" || channelString == "C4CR")
        {
            std::stringstream ss;
            ss << "    if (aSrcChannel >= 4)" << std::endl;
            ss << "    {" << std::endl;
            ss << R"(        throw INVALIDARGUMENT(aSrcChannel, "Value must be in range [0..3] but provided value is: " << aSrcChannel);)"
               << std::endl;
            ss << "    }" << std::endl;
            mCallHeader = ss.str();
        }
    }
    if (IsCopyInsert() || IsCopyChannel() || IsSetChannel())
    {
        // Add additional channel argument:
        mArguments.emplace_back(*this, "size_t", "aDstChannel");

        const std::string channelString = NPPParser::GetChannelString(aFunction.name);

        if (channelString == "C1C2R")
        {
            std::stringstream ss;
            ss << "    if (aDstChannel >= 2)" << std::endl;
            ss << "    {" << std::endl;
            ss << R"(        throw INVALIDARGUMENT(aDstChannel, "Value must be in range [0..1] but provided value is: " << aDstChannel);)"
               << std::endl;
            ss << "    }" << std::endl;
            mCallHeader = ss.str();
        }
        if (channelString == "C1C3R" || channelString == "C3CR")
        {
            std::stringstream ss;
            ss << "    if (aDstChannel >= 3)" << std::endl;
            ss << "    {" << std::endl;
            ss << R"(        throw INVALIDARGUMENT(aDstChannel, "Value must be in range [0..2] but provided value is: " << aDstChannel);)"
               << std::endl;
            ss << "    }" << std::endl;
            mCallHeader = ss.str();
        }
        if (channelString == "C1C4R" || channelString == "C4CR")
        {
            std::stringstream ss;
            ss << "    if (aDstChannel >= 4)" << std::endl;
            ss << "    {" << std::endl;
            ss << R"(        throw INVALIDARGUMENT(aDstChannel, "Value must be in range [0..3] but provided value is: " << aDstChannel);)"
               << std::endl;
            ss << "    }" << std::endl;
            mCallHeader = ss.str();
        }
    }

    // all planar input functions are made static:
    mIsStatic = IsInputPlanar();

    mIsConst = !IsInplace() && !IsMissingConst() && !mIsStatic;

    mIsMasked = NPPParser::IsMasked(mFunction);

    if (IsInputPlanar())
    {
        bool addParams = false;
        // check that input is given as array, otherwise we keep the original arguments:
        for (const auto &arg : mFunction.arguments)
        {
            // so far all planar inputs are name pSrc or aSrc (or pSrcY... but we want to ignore it anyhow)
            if (arg.name == "pSrc" || arg.name == "aSrc")
            {
                addParams = arg.type.find('[') != std::string::npos;
                break;
            }
        }

        if (addParams)
        {
            // add source image planes to argument list
            const int channelCount = InputPlanarCount();
            for (int i = 0; i < channelCount; i++)
            {
                const std::string type = "const Image" + GetTypeString() + "C1View &";
                mArguments.emplace_back(*this, type, "aSrcChannel" + std::to_string(i));
            }
        }
    }

    if (IsOutputPlanar())
    {
        bool addParams = false;
        for (const auto &arg : mFunction.arguments)
        {
            if (arg.name == "pDst" || arg.name == "aDst")
            {
                addParams = arg.type.find('[') != std::string::npos;
                break;
            }
        }

        if (addParams)
        {
            // add destination image planes to argument list
            const int channelCount = OutputPlanarCount();
            for (int i = 0; i < channelCount; i++)
            {
                const std::string type = "Image" + GetTypeString() + "C1View &";
                mArguments.emplace_back(*this, type, "aDstChannel" + std::to_string(i));
            }
        }
    }

    if (IsInputPlanar() && IsOutputPlanar())
    {
        // case when inplace planar
        bool addParams = false;
        // check that input is given as array, otherwise we keep the original arguments:
        for (const auto &arg : mFunction.arguments)
        {
            if (arg.name == "pSrcDst")
            {
                addParams = arg.type.find('[') != std::string::npos;
                break;
            }
        }

        if (addParams)
        {
            // add source image planes to argument list
            const int channelCount = InputPlanarCount();
            for (int i = 0; i < channelCount; i++)
            {
                const std::string type = "Image" + GetTypeString() + "C1View &";
                mArguments.emplace_back(*this, type, "aSrcDstChannel" + std::to_string(i));
            }
        }
    }

    for (const auto &elem : aFunction.arguments)
    {
        mArguments.emplace_back(&elem, *this);
    }

    if (IsFullROIFunction())
    {
        mArguments.emplace_back(*this, "const Roi &", "aFilterArea");
        std::stringstream ss;
        ss << "    NppiSize filterSize{aFilterArea.width, aFilterArea.height};" << std::endl;
        ss << "    const NppiPoint filterOffset{aFilterArea.x, aFilterArea.y};" << std::endl;
        ss << "    if (aFilterArea.Size() == Size2D())" << std::endl;
        ss << "    {" << std::endl;
        ss << "        filterSize.width  = ROI().width;" << std::endl;
        ss << "        filterSize.height = ROI().height;" << std::endl;
        ss << "    }" << std::endl;

        mCallHeader += ss.str();
    }

    const bool isBufferSize = IsGetBufferSizeFunction();
    if (isBufferSize)
    {
        mReturnType = "size_t";

        std::stringstream ssHeader;
        std::stringstream ssFooter;
        bool isUnsigned = false;
        bool isSizeT    = false;
        for (auto &arg : mArguments)
        {
            if (arg.Name().find("BufferSize") != std::string::npos)
            {
                arg.Call() = "&retValue";
                if (arg.Type().find('u') != std::string::npos)
                {
                    isUnsigned = true;
                    break;
                }
                if (arg.Type().find("size_t") != std::string::npos)
                {
                    isSizeT = true;
                    break;
                }
            }
        }

        if (isUnsigned)
        {
            ssHeader << "    uint retValue = 0;" << std::endl;
            ssFooter << "    return to_size_t(retValue);" << std::endl;
        }
        else if (isSizeT)
        {
            ssHeader << "    size_t retValue = 0;" << std::endl;
            ssFooter << "    return retValue;" << std::endl;
        }
        else
        {
            ssHeader << "    int retValue = 0;" << std::endl;
            ssFooter << "    return to_size_t(retValue);" << std::endl;
        }
        mCallHeader += ssHeader.str();
        mCallFooter += ssFooter.str();
    }

    if (mIsMasked && isBufferSize)
    {
        mName += "Masked"; // otherwise it would conflict with non masked version
    }

    if (IsAlphaIgnored() && mName[mName.size() - 1] != 'A')
    {
        mName += 'A'; // otherwise it would conflict with non alpha version
    }

    if (sAddFloatToName.contains(mFunction.name))
    {
        mName += "Float"; // otherwise it would conflict with non-float version
    }

    // link arguments:
    for (size_t i = 0; i < mArguments.size(); i++)
    {
        if (!mArguments[i].IsInputImage() && !mArguments[i].IsOutputImage() && !mArguments[i].IsMask())
        {
            continue;
        }
        for (size_t lookahead = 1; lookahead < 3 && (i + lookahead < mArguments.size()); lookahead++)
        {
            if (mArguments[i + lookahead].IsStep())
            {
                mArguments[i].LinkedArgument()             = &mArguments[i + lookahead];
                mArguments[i + lookahead].LinkedArgument() = &mArguments[i];
            }
        }
    }

    auto inputImages  = GetInputImages();
    auto outputImages = GetOutputImages();

    if (IsInputPlanar())
    {
        if (inputImages.size() == 1 || mName == "Remap")
        {
            const int channelCount = InputPlanarCount();
            std::stringstream ss;
            std::string type = inputImages[0]->SrcArgument()->type;
            type             = type.substr(0, type.find('['));
            size_t posConst  = type.find("const");
            while (posConst != std::string::npos)
            {
                type.replace(posConst, 5, "");
                posConst = type.find("const", posConst);
            }
            if (type[0] == ' ')
            {
                type = type.substr(1);
            }

            ss << "    const " << type << " srcList[] = { ";
            for (int i = 0; i < channelCount; i++)
            {
                ss << "reinterpret_cast<const " << type << ">(aSrcChannel" << i;
                if (IsGeometryFunction())
                {
                    ss << ".Pointer())";
                }
                else
                {
                    ss << ".PointerRoi())";
                }
                if (i != channelCount - 1)
                {
                    ss << ", ";
                }
                else
                {
                    ss << " };" << std::endl;
                }
            }
            mCallHeader += ss.str();
            inputImages[0]->Call() = "srcList";

            if (inputImages[0]->LinkedArgument()->SrcArgument()->type.find('[') != std::string::npos)
            {
                std::stringstream ss2;
                ss2 << "    int pitchSrcList[] = { ";
                for (int i = 0; i < channelCount; i++)
                {
                    ss2 << "to_int(aSrcChannel" << i << ".Pitch())";
                    if (i != channelCount - 1)
                    {
                        ss2 << ", ";
                    }
                    else
                    {
                        ss2 << " };" << std::endl;
                    }
                }
                mCallHeader += ss2.str();
                inputImages[0]->LinkedArgument()->Call() = "pitchSrcList";
            }
            else
            {
                std::stringstream ss2;
                // make sure that actually all pitch are the same:
                for (int i = 1; i < channelCount; i++)
                {
                    ss2 << "    if (aSrcChannel0.Pitch() != aSrcChannel" << i << ".Pitch())" << std::endl;
                    ss2 << "    {" << std::endl;
                    ss2 << "        throw INVALIDARGUMENT(aSrcChannel" << i
                        << ", \"Not all source image planes have the same image pitch. First image pitch is \" << "
                           "aSrcChannel0.Pitch() << \" but the pitch for plane no "
                        << i << " is \" << aSrcChannel" << i << ".Pitch()"
                        << ");" << std::endl;
                    ss2 << "    }" << std::endl;
                }
                mCallHeader += ss2.str();
                inputImages[0]->LinkedArgument()->Call() = "to_int(aSrcChannel0.Pitch())";
            }
        }

        if (inputImages.empty() && IsInplace())
        {
            const int channelCount = InputPlanarCount();
            std::stringstream ss;
            std::string type = mArguments[to_size_t(channelCount)].SrcArgument()->type;
            type             = type.substr(0, type.find('['));
            size_t posConst  = type.find("const");
            while (posConst != std::string::npos)
            {
                type.replace(posConst, 5, "");
                posConst = type.find("const", posConst);
            }
            if (type[0] == ' ')
            {
                type = type.substr(1);
            }

            ss << "    " << type << " srcList[] = { ";
            for (int i = 0; i < channelCount; i++)
            {
                ss << "reinterpret_cast<" << type << ">(aSrcDstChannel" << i;
                if (IsGeometryFunction())
                {
                    ss << ".Pointer())";
                }
                else
                {
                    ss << ".PointerRoi())";
                }
                if (i != channelCount - 1)
                {
                    ss << ", ";
                }
                else
                {
                    ss << " };" << std::endl;
                }
            }
            mCallHeader += ss.str();
            mArguments[to_size_t(channelCount)].Call() = "srcList";

            if (mArguments[to_size_t(channelCount) + 1].SrcArgument()->type.find('[') != std::string::npos)
            {
                std::stringstream ss2;
                ss2 << "    int pitchSrcList[] = { ";
                for (int i = 0; i < channelCount; i++)
                {
                    ss2 << "to_int(aSrcDstChannel" << i << ".Pitch())";
                    if (i != channelCount - 1)
                    {
                        ss2 << ", ";
                    }
                    else
                    {
                        ss2 << " };" << std::endl;
                    }
                }
                mCallHeader += ss2.str();
                mArguments[to_size_t(channelCount) + 1].Call() = "pitchSrcList";
            }
            else
            {
                std::stringstream ss2;
                // make sure that actually all pitch are the same:
                for (int i = 1; i < channelCount; i++)
                {
                    ss2 << "    if (aSrcDstChannel0.Pitch() != aSrcDstChannel" << i << ".Pitch())" << std::endl;
                    ss2 << "    {" << std::endl;
                    ss2 << "        throw INVALIDARGUMENT(aSrcDstChannel" << i
                        << ", \"Not all source image planes have the same image pitch. First image pitch is \" << "
                           "aSrcDstChannel0.Pitch() << \" but the pitch for plane no "
                        << i << " is \" << aSrcDstChannel" << i << ".Pitch()"
                        << ");" << std::endl;
                    ss2 << "    }" << std::endl;
                }
                mCallHeader += ss2.str();
                mArguments[to_size_t(channelCount) + 1].Call() = "to_int(aSrcDstChannel0.Pitch())";
            }
        }
    }

    if (IsOutputPlanar())
    {
        if (outputImages.size() == 1)
        {
            outputImages[0]->IsSkippedInDeclaration() = true;
            const int channelCount                    = OutputPlanarCount();
            std::stringstream ss;
            std::string type = outputImages[0]->SrcArgument()->type;
            type             = type.substr(0, type.find('['));
            size_t posConst  = type.find("const");
            while (posConst != std::string::npos)
            {
                type.replace(posConst, 5, "");
                posConst = type.find("const", posConst);
            }
            if (type[0] == ' ')
            {
                type = type.substr(1);
            }

            ss << "    " << type << " dstList[] = { ";
            for (int i = 0; i < channelCount; i++)
            {
                ss << "reinterpret_cast<" << type << ">(aDstChannel" << i << ".PointerRoi())";
                if (i != channelCount - 1)
                {
                    ss << ", ";
                }
                else
                {
                    ss << " };" << std::endl;
                }
            }
            mCallHeader += ss.str();
            outputImages[0]->Call() = "dstList";

            if (outputImages[0]->LinkedArgument()->SrcArgument()->type.find('[') != std::string::npos)
            {
                std::stringstream ss2;
                ss2 << "    int pitchDstList[] = { ";
                for (int i = 0; i < channelCount; i++)
                {
                    ss2 << "to_int(aDstChannel" << i << ".Pitch())";
                    if (i != channelCount - 1)
                    {
                        ss2 << ", ";
                    }
                    else
                    {
                        ss2 << " };" << std::endl;
                    }
                }
                mCallHeader += ss2.str();
                outputImages[0]->LinkedArgument()->Call() = "pitchDstList";
            }
            else
            {
                std::stringstream ss2;
                // make sure that actually all pitch are the same:
                for (int i = 1; i < channelCount; i++)
                {
                    ss2 << "    if (aDstChannel0.Pitch() != aDstChannel" << i << ".Pitch())" << std::endl;
                    ss2 << "    {" << std::endl;
                    ss2 << "        throw INVALIDARGUMENT(aDstChannel" << i
                        << ", \"Not all destination image planes have the same image pitch. First image pitch is \" << "
                           "aDstChannel0.Pitch() << \" but the pitch for plane no "
                        << i << " is \" << aDstChannel" << i << ".Pitch()"
                        << ");" << std::endl;
                    ss2 << "    }" << std::endl;
                }
                mCallHeader += ss2.str();
                outputImages[0]->LinkedArgument()->Call() = "to_int(aDstChannel0.Pitch())";
            }
        }
    }

    // size ROI checker
    {
        std::stringstream ss;

        if (!IsDistanceMeasureFuntion() && !IsGeometryFunction() && !IsInputPlanar() && !IsOutputPlanar() &&
            mName.find("Pyramid") == std::string::npos && mName.find("Integral") == std::string::npos &&
            mName != "CopyWrapBorder" && mName != "CopyReplicateBorder" && mName != "CopyConstBorder" &&
            mName != "CopyWrapBorderA" && mName != "CopyReplicateBorderA" && mName != "CopyConstBorderA")
        {
            // If it is a geometry function, the ROIs may differ
            // in the planar case we can have different ROIs due to different samplings

            for (auto *inputImage : inputImages)
            {
                if (!inputImage->IsSkippedInDeclaration() && inputImage->Name() != "pTpl")
                {
                    ss << "    checkSameSize(ROI(), " << inputImage->Name() << ".ROI());" << std::endl;
                }
            }
            for (auto *outputImage : outputImages)
            {
                if (!outputImage->IsSkippedInDeclaration() && mName != "Transpose")
                {
                    ss << "    checkSameSize(ROI(), " << outputImage->Name() << ".ROI());" << std::endl;
                }
                else if (!outputImage->IsSkippedInDeclaration() && mName == "Transpose")
                {
                    ss << "    checkSameSize(ROI().Size(), Size2D(pDst.SizeRoi().y, pDst.SizeRoi().x));" << std::endl;
                }
            }
        }
        else if (mName == "Remap" && !IsOutputPlanar())
        {
            ss << "    checkSameSize(pXMap.ROI(), pDst.ROI());" << std::endl;
            ss << "    checkSameSize(pYMap.ROI(), pDst.ROI());" << std::endl;
        }
        else if (mName == "Remap" && IsOutputPlanar())
        {
            ss << "    checkSameSize(pXMap.ROI(), aDstChannel0.ROI());" << std::endl;
            ss << "    checkSameSize(pYMap.ROI(), aDstChannel0.ROI());" << std::endl;
        }
        else if (mName == "Integral")
        {
            ss << "    checkSameSize(ROI().Size(), pDst.ROI().Size() - 1);" << std::endl;
        }
        else if (mName == "SqrIntegral")
        {
            ss << "    checkSameSize(ROI().Size(), pDst.ROI().Size() - 1);" << std::endl;
            ss << "    checkSameSize(ROI().Size(), pSqr.ROI().Size() - 1);" << std::endl;
        }
        mCallHeader += ss.str();
    }

    // set call for input images
    {
        for (auto *inputImage : inputImages)
        {
            if ((inputImages.size() == 1 || (mName == "Remap" && inputImage->SrcArgument()->name == "pSrc")) &&
                IsInputPlanar())
            {
                // this is already set...
            }
            else if (inputImage->IsSkippedInDeclaration() && (IsGeometryFunction() || IsCFAToRGBFunction()))
            {
                inputImage->Call() =
                    std::string("reinterpret_cast<") + inputImage->SrcArgument()->type + ">(Pointer())";
                inputImage->LinkedArgument()->Call() = "to_int(Pitch())";
            }
            else if (inputImage->IsSkippedInDeclaration() && (IsCopyExtract() || IsCopyChannel()))
            {
                inputImage->Call() = std::string("reinterpret_cast<") + inputImage->SrcArgument()->type +
                                     ">(PointerRoi()) + aSrcChannel";
                inputImage->LinkedArgument()->Call() = "to_int(Pitch())";
            }
            else if (inputImage->IsSkippedInDeclaration())
            {
                inputImage->Call() =
                    std::string("reinterpret_cast<") + inputImage->SrcArgument()->type + ">(PointerRoi())";
                inputImage->LinkedArgument()->Call() = "to_int(Pitch())";
            }
            else if (IsCopyExtract() || IsCopyChannel())
            {
                inputImage->Call() = std::string("reinterpret_cast<") + inputImage->SrcArgument()->type + ">(" +
                                     inputImage->Name() + ".PointerRoi()) + aSrcChannel";
                inputImage->LinkedArgument()->Call() = std::string("to_int(") + inputImage->Name() + ".Pitch())";
            }
            else
            {
                std::string type = inputImage->SrcArgument()->type;
                if (type == "const Npp8u *const")
                {
                    type = "const Npp8u *";
                }
                inputImage->Call() =
                    std::string("reinterpret_cast<") + type + ">(" + inputImage->Name() + ".PointerRoi())";
                inputImage->LinkedArgument()->Call() = std::string("to_int(") + inputImage->Name() + ".Pitch())";
            }
        }
    }

    // set call for output images
    {
        for (auto *outputImage : outputImages)
        {
            if (outputImages.size() == 1 && IsOutputPlanar())
            {
                // this is already set...
            }
            else if (outputImage->IsSkippedInDeclaration() && IsSetChannel())
            {
                outputImage->Call() = std::string("reinterpret_cast<") + outputImage->SrcArgument()->type +
                                      ">(PointerRoi()) + aDstChannel";
                outputImage->LinkedArgument()->Call() = "to_int(Pitch())";
            }
            else if (outputImage->IsSkippedInDeclaration())
            {
                outputImage->Call() =
                    std::string("reinterpret_cast<") + outputImage->SrcArgument()->type + ">(PointerRoi())";
                outputImage->LinkedArgument()->Call() = "to_int(Pitch())";
            }
            else if (IsCopyInsert() || IsCopyChannel())
            {
                outputImage->Call() = std::string("reinterpret_cast<") + outputImage->SrcArgument()->type + ">(" +
                                      outputImage->Name() + ".PointerRoi()) + aDstChannel";
                outputImage->LinkedArgument()->Call() = std::string("to_int(") + outputImage->Name() + ".Pitch())";
            }
            else if (IsGeometryFunction())
            {
                outputImage->Call() = std::string("reinterpret_cast<") + outputImage->SrcArgument()->type + ">(" +
                                      outputImage->Name() + ".Pointer())";
                outputImage->LinkedArgument()->Call() = std::string("to_int(") + outputImage->Name() + ".Pitch())";
            }
            else
            {
                outputImage->Call() = std::string("reinterpret_cast<") + outputImage->SrcArgument()->type + ">(" +
                                      outputImage->Name() + ".PointerRoi())";
                outputImage->LinkedArgument()->Call() = std::string("to_int(") + outputImage->Name() + ".Pitch())";
            }
        }
    }

    // set call for mask images
    {
        for (auto &mask : mArguments)
        {
            if (mask.IsMask())
            {
                mask.Call() =
                    std::string("reinterpret_cast<") + mask.SrcArgument()->type + ">(" + mask.Name() + ".PointerRoi())";
                mask.LinkedArgument()->Call() = std::string("to_int(") + mask.Name() + ".Pitch())";
            }
        }
    }

    // set call for ROI and size parameters:
    {
        for (auto &arg : mArguments)
        {
            if (inputImages.empty() && IsInputPlanar() &&
                (arg.Name() == "oSizeROI" || arg.Name() == "oSrcSizeROI" || arg.Name() == "oSrcRoiSize" ||
                 arg.Name() == "oROI" || (arg.Name() == "oSrcROI" && mName == "Transpose") ||
                 (arg.Name() == "oSrcROI" && mName == "SqrIntegral") || (arg.Name() == "oSrcROI")))
            {
                if (arg.SrcArgument()->type == "NppiSize")
                {
                    arg.Call() = "aSrcDstChannel0.NppiSizeRoi()";
                }
                else
                {
                    arg.Call() = "aSrcDstChannel0.NppiRectRoi()";
                }
            }
            else if (inputImages.size() == 1 && IsInputPlanar() &&
                     (arg.Name() == "oSizeROI" || arg.Name() == "oSrcSizeROI" || arg.Name() == "oSrcRoiSize" ||
                      arg.Name() == "oROI" || (arg.Name() == "oSrcROI" && mName == "Transpose") ||
                      (arg.Name() == "oSrcROI" && mName == "SqrIntegral") || (arg.Name() == "oSrcROI")))
            {
                if (arg.SrcArgument()->type == "NppiSize")
                {
                    arg.Call() = "aSrcChannel0.NppiSizeRoi()";
                }
                else
                {
                    arg.Call() = "aSrcChannel0.NppiRectRoi()";
                }
            }
            else if (inputImages.size() == 2 && IsInputPlanar() && mIsStatic && (arg.Name() == "oSizeROI"))
            {
                arg.Call() = "pSrcY.NppiSizeRoi()";
            }
            else if (arg.Name() == "oSizeROI" || arg.Name() == "oSrcSizeROI" || arg.Name() == "oSrcRoiSize" ||
                     arg.Name() == "oROI" || (arg.Name() == "oSrcROI" && mName == "Transpose") ||
                     (arg.Name() == "oSrcROI" && mName == "SqrIntegral") ||
                     (arg.Name() == "oSrcROI" && mName == "ResizeAdvancedGetBufferHostSize"))
            {
                arg.Call() = "NppiSizeRoi()";
            }
            else if (arg.Name() == "oTplRoiSize")
            {
                arg.Call() = "pTpl.NppiSizeRoi()";
            }
            else if (arg.Name() == "oSrcSize" && IsFullROIFunction())
            {
                arg.Call() = "filterSize";
            }
            else if (arg.Name() == "oSrcOffset" && IsFullROIFunction())
            {
                arg.Call() = "filterOffset";
            }
            else if (arg.Name() == "oSrcOffset" && (mName == "FilterUnsharpBorder" || mName == "FilterUnsharpBorderA"))
            {
                arg.Call() = "NppiPointRoi()";
            }
            else if (arg.Name() == "oSrcSize" && !IsInputPlanar())
            {
                arg.Call() = "NppiSizeFull()";
            }
            else if (arg.Name() == "oSrcSize" && IsInputPlanar())
            {
                arg.Call() = "aSrcChannel0.NppiSizeFull()";
            }
            else if (arg.Name() == "oDstSize" && !IsOutputPlanar())
            {
                arg.Call() = "pDst.NppiSizeFull()";
            }
            else if (arg.Name() == "oDstSize" && IsOutputPlanar())
            {
                arg.Call() = "aDstChannel0.NppiSizeFull()";
            }
            else if (((arg.Name() == "oSrcROI" && mName != "Transpose" && mName != "SqrIntegral") ||
                      arg.Name() == "oSrcRectROI") &&
                     !IsInputPlanar())
            {
                arg.Call() = "NppiRectRoi()";
            }
            else if (((arg.Name() == "oSrcROI" && mName != "Transpose" && mName != "SqrIntegral") ||
                      arg.Name() == "oSrcRectROI") &&
                     IsInputPlanar())
            {
                arg.Call() = "aSrcChannel0.NppiRectRoi()";
            }
            else if (arg.Name() == "oDstSizeROI" && !IsOutputPlanar())
            {
                arg.Call() = "pDst.NppiSizeRoi()";
            }
            else if (arg.Name() == "oDstSizeROI" && IsOutputPlanar())
            {
                arg.Call() = "aDstChannel0.NppiSizeRoi()";
            }
            else if ((arg.Name() == "oDstROI" && mName == "ResizeAdvancedGetBufferHostSize"))
            {
                arg.Call() = "oDstROI.NppiSizeRoi()";
            }
            else if (((arg.Name() == "oDstROI" && arg.Type().find("NppPointPolar") == std::string::npos) ||
                      arg.Name() == "oDstRectROI") &&
                     !IsOutputPlanar())
            {
                arg.Call() = "pDst.NppiRectRoi()";
            }
            else if (((arg.Name() == "oDstROI" && arg.Type().find("NppPointPolar") == std::string::npos) ||
                      arg.Name() == "oDstRectROI") &&
                     IsOutputPlanar())
            {
                arg.Call() = "aDstChannel0.NppiRectRoi()";
            }
        }
    }

    // set call for Pixel values:
    {
        for (auto &arg : mArguments)
        {
            if (arg.Type() == "const Pixel8uC1 &" || arg.Type() == "const Pixel8sC1 &" ||
                arg.Type() == "const Pixel16uC1 &" || arg.Type() == "const Pixel16sC1 &" ||
                arg.Type() == "const Pixel32uC1 &" || arg.Type() == "const Pixel32sC1 &" ||
                arg.Type() == "const Pixel32fC1 &" || arg.Type() == "const Pixel64fC1 &" ||
                arg.Type() == "const Pixel16fC1 &" ||

                arg.Type() == "Pixel8uC1 &" || arg.Type() == "Pixel8sC1 &" || arg.Type() == "Pixel16uC1 &" ||
                arg.Type() == "Pixel16sC1 &" || arg.Type() == "Pixel32uC1 &" || arg.Type() == "Pixel32sC1 &" ||
                arg.Type() == "Pixel32fC1 &" || arg.Type() == "Pixel64fC1 &" || arg.Type() == "Pixel16fC1 &")
            {
                arg.Call() += ".x";
            }
            else if (arg.Type() == "const Pixel16scC1 &")
            {
                arg.Call() = "*reinterpret_cast<const Npp16sc *>(&" + arg.Call() + ")";
            }
            else if (arg.Type() == "const Pixel32scC1 &")
            {
                arg.Call() = "*reinterpret_cast<const Npp32sc *>(&" + arg.Call() + ")";
            }
            else if (arg.Type() == "const Pixel32fcC1 &")
            {
                arg.Call() = "*reinterpret_cast<const Npp32fc *>(&" + arg.Call() + ")";
            }
            else if (arg.Type() == "Pixel16scC1 &")
            {
                arg.Call() = "*reinterpret_cast<Npp16sc *>(&" + arg.Call() + ")";
            }
            else if (arg.Type() == "Pixel32scC1 &")
            {
                arg.Call() = "*reinterpret_cast<Npp32sc *>(&" + arg.Call() + ")";
            }
            else if (arg.Type() == "Pixel32fcC1 &")
            {
                arg.Call() = "*reinterpret_cast<Npp32fc *>(&" + arg.Call() + ")";
            }
            else if (arg.Type() == "const Pixel8uC2 &" || arg.Type() == "const Pixel8sC2 &" ||
                     arg.Type() == "const Pixel16uC2 &" || arg.Type() == "const Pixel16sC2 &" ||
                     arg.Type() == "const Pixel32uC2 &" || arg.Type() == "const Pixel32sC2 &" ||
                     arg.Type() == "const Pixel32fC2 &" || arg.Type() == "const Pixel64fC2 &" ||
                     arg.Type() == "const Pixel16fC2 &" || arg.Type() == "const Pixel8uC3 &" ||
                     arg.Type() == "const Pixel8sC3 &" || arg.Type() == "const Pixel16uC3 &" ||
                     arg.Type() == "const Pixel16sC3 &" || arg.Type() == "const Pixel32uC3 &" ||
                     arg.Type() == "const Pixel32sC3 &" || arg.Type() == "const Pixel32fC3 &" ||
                     arg.Type() == "const Pixel64fC3 &" || arg.Type() == "const Pixel16fC3 &" ||
                     arg.Type() == "const Pixel8uC4 &" || arg.Type() == "const Pixel8sC4 &" ||
                     arg.Type() == "const Pixel16uC4 &" || arg.Type() == "const Pixel16sC4 &" ||
                     arg.Type() == "const Pixel32uC4 &" || arg.Type() == "const Pixel32sC4 &" ||
                     arg.Type() == "const Pixel32fC4 &" || arg.Type() == "const Pixel64fC4 &" ||
                     arg.Type() == "const Pixel16fC4 &" || arg.Type() == "const Pixel8uC4A &" ||
                     arg.Type() == "const Pixel8sC4A &" || arg.Type() == "const Pixel16uC4A &" ||
                     arg.Type() == "const Pixel16sC4A &" || arg.Type() == "const Pixel32uC4A &" ||
                     arg.Type() == "const Pixel32sC4A &" || arg.Type() == "const Pixel32fC4A &" ||
                     arg.Type() == "const Pixel64fC4A &" || arg.Type() == "const Pixel16fC4A &" ||

                     arg.Type() == "Pixel8uC2 &" || arg.Type() == "Pixel8sC2 &" || arg.Type() == "Pixel16uC2 &" ||
                     arg.Type() == "Pixel16sC2 &" || arg.Type() == "Pixel32uC2 &" || arg.Type() == "Pixel32sC2 &" ||
                     arg.Type() == "Pixel32fC2 &" || arg.Type() == "Pixel64fC2 &" || arg.Type() == "Pixel16fC2 &" ||

                     arg.Type() == "Pixel8uC3 &" || arg.Type() == "Pixel8sC3 &" || arg.Type() == "Pixel16uC3 &" ||
                     arg.Type() == "Pixel16sC3 &" || arg.Type() == "Pixel32uC3 &" || arg.Type() == "Pixel32sC3 &" ||
                     arg.Type() == "Pixel32fC3 &" || arg.Type() == "Pixel64fC3 &" || arg.Type() == "Pixel16fC3 &" ||
                     arg.Type() == "Pixel8uC4 &" || arg.Type() == "Pixel8sC4 &" || arg.Type() == "Pixel16uC4 &" ||
                     arg.Type() == "Pixel16sC4 &" || arg.Type() == "Pixel32uC4 &" || arg.Type() == "Pixel32sC4 &" ||
                     arg.Type() == "Pixel32fC4 &" || arg.Type() == "Pixel64fC4 &" || arg.Type() == "Pixel16fC4 &" ||
                     arg.Type() == "Pixel8uC4A &" || arg.Type() == "Pixel8sC4A &" || arg.Type() == "Pixel16uC4A &" ||
                     arg.Type() == "Pixel16sC4A &" || arg.Type() == "Pixel32uC4A &" || arg.Type() == "Pixel32sC4A &" ||
                     arg.Type() == "Pixel32fC4A &" || arg.Type() == "Pixel64fC4A &" || arg.Type() == "Pixel16fC4A &")
            {
                arg.Call() += ".data()";
            }
            else if (arg.Type() == "const Pixel16scC2 &" || arg.Type() == "const Pixel16scC3 &" ||
                     arg.Type() == "const Pixel16scC4 &" || arg.Type() == "const Pixel16scC4A &")
            {
                arg.Call() = "reinterpret_cast<const Npp16sc *>(" + arg.Call() + ".data())";
            }
            else if (arg.Type() == "const Pixel32scC2 &" || arg.Type() == "const Pixel32scC3 &" ||
                     arg.Type() == "const Pixel32scC4 &" || arg.Type() == "const Pixel32scC4A &")
            {
                arg.Call() = "reinterpret_cast<const Npp32sc *>(" + arg.Call() + ".data())";
            }
            else if (arg.Type() == "const Pixel32fcC2 &" || arg.Type() == "const Pixel32fcC3 &" ||
                     arg.Type() == "const Pixel32fcC4 &" || arg.Type() == "const Pixel32fcC4A &")
            {
                arg.Call() = "reinterpret_cast<const Npp32fc *>(" + arg.Call() + ".data())";
            }
            else if (arg.Type() == "Pixel16scC2 &" || arg.Type() == "Pixel16scC3 &" || arg.Type() == "Pixel16scC4 &" ||
                     arg.Type() == "Pixel16scC4A &")
            {
                arg.Call() = "reinterpret_cast<Npp16sc *>(" + arg.Call() + ".data())";
            }
            else if (arg.Type() == "Pixel32scC2 &" || arg.Type() == "Pixel32scC3 &" || arg.Type() == "Pixel32scC4 &" ||
                     arg.Type() == "Pixel32scC4A &")
            {
                arg.Call() = "reinterpret_cast<Npp32sc *>(" + arg.Call() + ".data())";
            }
            else if (arg.Type() == "Pixel32fcC2 &" || arg.Type() == "Pixel32fcC3 &" || arg.Type() == "Pixel32fcC4 &" ||
                     arg.Type() == "Pixel32fcC4A &")
            {
                arg.Call() = "reinterpret_cast<Npp32fc *>(" + arg.Call() + ".data())";
            }
            else if (arg.Type().find("DevVarView") != std::string::npos && arg.Call() != "pLevelsPtrList" &&
                     arg.Call() != "pValuesPtrList" && arg.Call() != "pHistPtrList" && arg.Call() != "pTablesPtrList")
            {
                bool needCast = true;
                if (arg.Type().find("<sbyte>") != std::string::npos ||  //
                    arg.Type().find("<byte>") != std::string::npos ||   //
                    arg.Type().find("<short>") != std::string::npos ||  //
                    arg.Type().find("<ushort>") != std::string::npos || //
                    arg.Type().find("<uint>") != std::string::npos ||   //
                    arg.Type().find("<int>") != std::string::npos ||    //
                    // arg.Type().find("<long64>") != std::string::npos || //
                    arg.Type().find("<float>") != std::string::npos ||  //
                    arg.Type().find("<double>") != std::string::npos || //
                    arg.Type().find("<NppiPoint>") != std::string::npos)
                {
                    needCast = false;
                }

                if (needCast)
                {
                    const size_t pos = arg.SrcArgument()->type.find('[');
                    if (pos != std::string::npos)
                    {
                        const std::string type = arg.SrcArgument()->type.substr(0, pos);
                        arg.Call() = std::string("reinterpret_cast<") + type + " *>(" + arg.Name() + ".Pointer())";
                    }
                    else
                    {
                        arg.Call() = std::string("reinterpret_cast<") + arg.SrcArgument()->type + ">(" + arg.Name() +
                                     ".Pointer())";
                    }
                }
                else
                {
                    arg.Call() = arg.Name() + ".Pointer()";
                }
            }
        }
    }

    // set exception debug message:
    {
        if (!isBufferSize && !IsInputPlanar() && !IsOutputPlanar())
        {
            std::stringstream ss;
            // write ROI information to exception message for debugging
            if (outputImages.size() == 1 && inputImages.empty())
            {
                if (IsInplace())
                {
                    ss << "\"ROI SrcDst: \" << ROI()";
                }
                else
                {
                    ss << "\"ROI Dst: \" << " << outputImages[0]->SrcArgument()->name << ".ROI()";
                }
            }
            else if (outputImages.size() == 1 && inputImages.size() == 1)
            {
                if (IsInplace())
                {
                    ss << "\"ROI Src2: \" << " << inputImages[0]->SrcArgument()->name << ".ROI() << ";
                    ss << "\"ROI SrcDst: \" << ROI()";
                }
                else
                {
                    ss << "\"ROI Src: \" << ROI() << ";
                    ss << "\"ROI Dst: \" << " << outputImages[0]->SrcArgument()->name << ".ROI()";
                }
            }
            else if (outputImages.size() == 1 && inputImages.size() == 2)
            {
                if (IsInplace())
                {
                    ss << "\"ROI Src2: \" << " << inputImages[1]->SrcArgument()->name << ".ROI() << ";
                    ss << "\"ROI SrcDst: \" << ROI()";
                }
                else
                {
                    ss << R"("ROI Src1: " << ROI() << " ROI Src2: " << )" << inputImages[1]->SrcArgument()->name
                       << ".ROI() << ";
                    ss << "\"ROI Dst: \" << " << outputImages[0]->SrcArgument()->name << ".ROI()";
                }
            }
            else if (outputImages.size() == 2 && inputImages.empty())
            {
                if (mFunction.name != "nppiSegmentWatershed_8u_C1IR_Ctx" &&
                    mFunction.name != "nppiSegmentWatershed_16u_C1IR_Ctx")
                {
                    ss << "Failed log for nppiSegmentWatershed_8u_C1IR_Ctx";
                }
                // for SegmentWatershed
                ss << R"("ROI SrcDst: " << ROI() << " ROI MarkerLabels: " << )" << outputImages[1]->SrcArgument()->name
                   << ".ROI()";
            }
            else if (outputImages.empty() && inputImages.size() == 1)
            {
                // for statistics
                ss << "\"ROI Src: \" << ROI()";
            }
            else if (outputImages.empty() && inputImages.size() == 2)
            {
                // for statistics
                ss << R"("ROI Src1: " << ROI() << " ROI Src2: " << )" << inputImages[1]->SrcArgument()->name
                   << ".ROI()";
            }
            else if (outputImages.size() == 1 && inputImages.size() == 3)
            {
                // for Remap
                ss << R"("ROI Src: " << ROI() << " ROI pXMap: " << )" << inputImages[1]->SrcArgument()->name
                   << R"(.ROI() << " ROI pYMap: " << )" << inputImages[2]->SrcArgument()->name << ".ROI() << ";
                ss << "\"ROI Dst: \" << " << outputImages[0]->SrcArgument()->name << ".ROI()";
            }
            else if (outputImages.size() == 2 && inputImages.size() == 1)
            {
                ss << R"("ROI Src: " << ROI() << )";
                ss << R"("ROI )" << outputImages[0]->SrcArgument()->name << R"(: " << )"
                   << outputImages[0]->SrcArgument()->name << ".ROI() << ";
                ss << R"("ROI )" << outputImages[1]->SrcArgument()->name << R"(: " << )"
                   << outputImages[1]->SrcArgument()->name << ".ROI()";
            }
            else if (outputImages.size() == 4 && inputImages.size() == 1)
            {
                ss << R"("ROI Src: " << ROI() << )";
                ss << R"("ROI )" << outputImages[0]->SrcArgument()->name << R"(: " << )"
                   << outputImages[0]->SrcArgument()->name << ".ROI() << ";
                ss << R"("ROI )" << outputImages[1]->SrcArgument()->name << R"(: " << )"
                   << outputImages[1]->SrcArgument()->name << ".ROI() << ";
                ss << R"("ROI )" << outputImages[2]->SrcArgument()->name << R"(: " << )"
                   << outputImages[2]->SrcArgument()->name << ".ROI() << ";
                ss << R"("ROI )" << outputImages[3]->SrcArgument()->name << R"(: " << )"
                   << outputImages[3]->SrcArgument()->name << ".ROI()";
            }
            else
            {
                ss << "Failed log for " << mFunction.name;
                // aFailedFunctions.push_back(elem);
            }
            mExceptionMessage = ss.str();
        }
    }

    // set call header for pLevels, pValues and pHist
    {
        for (auto &arg : mArguments)
        {
            if (arg.Call() == "pLevelsPtrList")
            {
                const int channelCount = GetOutChannels();
                std::stringstream ss;
                ss << "    ";
                if (arg.Type().find("float") != std::string::npos)
                {
                    ss << "const float *pLevelsPtrList[] = { ";
                }
                else
                {
                    ss << "const int *pLevelsPtrList[] = { ";
                }
                for (int i = 0; i < channelCount; i++)
                {
                    ss << "pLevels[" << i << "].Pointer()";
                    if (i != channelCount - 1)
                    {
                        ss << ", ";
                    }
                    else
                    {
                        ss << " };" << std::endl;
                    }
                }
                mCallHeader += ss.str();
            }

            if (arg.Call() == "pValuesPtrList")
            {
                const int channelCount = GetOutChannels();
                std::stringstream ss;
                ss << "    ";
                if (arg.Type().find("float") != std::string::npos)
                {
                    ss << "const float *pValuesPtrList[] = { ";
                }
                else
                {
                    ss << "const int *pValuesPtrList[] = { ";
                }
                for (int i = 0; i < channelCount; i++)
                {
                    ss << "pValues[" << i << "].Pointer()";
                    if (i != channelCount - 1)
                    {
                        ss << ", ";
                    }
                    else
                    {
                        ss << " };" << std::endl;
                    }
                }
                mCallHeader += ss.str();
            }

            if (arg.Call() == "pHistPtrList")
            {
                const int channelCount = GetOutChannels();
                std::stringstream ss;
                ss << "    ";
                if (arg.Type().find("float") != std::string::npos)
                {
                    ss << "float *pHistPtrList[] = { ";
                }
                else
                {
                    ss << "int *pHistPtrList[] = { ";
                }
                for (int i = 0; i < channelCount; i++)
                {
                    ss << "pHist[" << i << "].Pointer()";
                    if (i != channelCount - 1)
                    {
                        ss << ", ";
                    }
                    else
                    {
                        ss << " };" << std::endl;
                    }
                }
                mCallHeader += ss.str();
            }

            if (arg.Call() == "pTablesPtrList")
            {
                const int channelCount = GetOutChannels();
                std::stringstream ss;
                ss << "    ";
                if (arg.Type().find("byte") != std::string::npos)
                {
                    ss << "const byte *pTablesPtrList[] = { ";
                }
                else
                {
                    ss << "const ushort *pTablesPtrList[] = { ";
                }
                for (int i = 0; i < channelCount; i++)
                {
                    ss << "pTables[" << i << "].Pointer()";
                    if (i != channelCount - 1)
                    {
                        ss << ", ";
                    }
                    else
                    {
                        ss << " };" << std::endl;
                    }
                }
                mCallHeader += ss.str();
            }
        }
    }

    mImageViewType = GetImageViewType();

    if (mFunction.name == "nppiSumGetBufferHostSize_8u_C4R_Ctx")
    {
        mName += "Int64";
    }

    if (outputImages.size() == 1 && mReturnType == "void")
    {
        // if we only have only one output image, use that as return value
        if (outputImages[0]->Name() == "pDst" && !outputImages[0]->IsSkippedInDeclaration() &&
            outputImages[0]->Type()[outputImages[0]->Type().size() - 1] == '&')
        {
            mReturnType = outputImages[0]->Type();
            std::stringstream ss;
            ss << "    return pDst;" << std::endl;
            mCallFooter += ss.str();
        }

        if ((outputImages[0]->Name() == "pSrcDst" || outputImages[0]->Name() == "pDst") &&
            outputImages[0]->IsSkippedInDeclaration() &&
            outputImages[0]->Type()[outputImages[0]->Type().size() - 1] == '&')
        {
            mReturnType = outputImages[0]->Type();
            std::stringstream ss;
            ss << "    return *this;" << std::endl;
            mCallFooter += ss.str();
        }
    }
}
std::string ConvertedFunction::ToStringHeader() const
{
    std::stringstream ss;

    ss << std::endl << "    // " << mFunction.returnType << " " << mFunction.name << "(";

    bool isFirst = true;
    for (const auto &arg : mArguments)
    {
        if (arg.SrcArgument() != nullptr)
        {
            if (!isFirst)
            {
                ss << ", ";
            }
            isFirst = false;
            ss << arg.SrcArgument()->type << " " << arg.SrcArgument()->name;
        }
    }
    ss << ")" << std::endl;

    ss << "    ";
    if (mIsStatic)
    {
        ss << "static ";
    }
    if (mReturnType != "void" && mReturnType[mReturnType.size() - 1] != '&')
    {
        ss << "[[nodiscard]] ";
    }
    ss << mReturnType;
    if (mReturnType[mReturnType.size() - 1] != '&')
    {
        ss << " ";
    }
    ss << mName << "(";
    bool isFirstArg = true;
    for (const auto &arg : mArguments)
    {
        if (arg.IsSkippedInDeclaration())
        {
            continue;
        }

        if (!isFirstArg)
        {
            ss << ", ";
        }
        isFirstArg = false;
        ss << arg.ToStringDeclaration();

        // add default value for aFilterArea
        if (arg.Name() == "aFilterArea")
        {
            ss << " = Roi()";
        }
    }
    ss << ")";
    if (mIsConst)
    {
        ss << " const";
    }
    ss << ";";

    return ss.str();
}
std::string ConvertedFunction::ToStringCpp() const
{
    std::stringstream ss;

    ss << mReturnType;
    if (mReturnType[mReturnType.size() - 1] != '&')
    {
        ss << " ";
    }
    ss << mImageViewType << "::" << mName << "(";
    bool isFirstArg = true;
    for (const auto &arg : mArguments)
    {
        if (arg.IsSkippedInDeclaration())
        {
            continue;
        }

        if (!isFirstArg)
        {
            ss << ", ";
        }
        isFirstArg = false;
        ss << arg.ToStringDeclaration();
    }
    ss << ")";
    if (mIsConst)
    {
        ss << " const";
    }
    ss << std::endl;
    ss << "{" << std::endl;
    ss << mCallHeader;
    if (mExceptionMessage.empty())
    {
        ss << "    nppSafeCall(" << mFunction.name << "(";
    }
    else
    {
        ss << "    nppSafeCallExt(" << mFunction.name << "(";
    }

    isFirstArg = true;
    for (const auto &arg : mArguments)
    {
        if (arg.IsSkippedInCall())
        {
            continue;
        }

        if (!isFirstArg)
        {
            ss << ", ";
        }
        isFirstArg = false;
        ss << arg.Call();
    }

    if (mExceptionMessage.empty())
    {
        ss << "));" << std::endl;
    }
    else
    {
        ss << ")," << std::endl;
        ss << "                   " << mExceptionMessage << ");" << std::endl;
    }

    ss << mCallFooter;

    ss << "}" << std::endl;

    return ss.str();
}

bool ConvertedFunction::IsMissingConst() const
{
    return sMissingConst.contains(mName);
}

bool ConvertedFunction::IsCopyInsert() const
{
    return sCopyInsert.contains(mFunction.name);
}

bool ConvertedFunction::IsCopyExtract() const
{
    return sCopyExtract.contains(mFunction.name);
}

bool ConvertedFunction::IsCopyChannel() const
{
    return sCopyChannel.contains(mFunction.name);
}

bool ConvertedFunction::IsSetChannel() const
{
    return sSetChannel.contains(mFunction.name);
}

bool ConvertedFunction::IsInplace() const
{
    return NPPParser::IsInplace(mFunction);
}

bool ConvertedFunction::IsGetBufferSizeFunction() const
{
    if (mFunction.name.find("GetBufferSize") != std::string::npos)
    {
        return true;
    }
    if (mFunction.name == "nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R")
    {
        return true;
    }

    return mFunction.name.find("GetBufferHostSize") != std::string::npos;
}

bool ConvertedFunction::IsFullROIFunction() const
{
    if (IsGetBufferSizeFunction())
    {
        return false;
    }

    if (mName == "CopyConstBorder" || mName == "CopyReplicateBorder" || mName == "CopyWrapBorder" ||
        mName == "FilterUnsharpBorder" || mName == "FilterUnsharpBorderA")
    {
        return false;
    }

    return mName.find("Border") != std::string::npos;
}

bool ConvertedFunction::IsGeometryFunction() const
{
    return sGeometryWithAllocPointer.contains(mName);
}

bool ConvertedFunction::IsCFAToRGBFunction() const
{
    return sCFAToRGBWithAllocPointer.contains(mName);
}

bool ConvertedFunction::IsDistanceMeasureFuntion() const
{
    return sDistanceMeasureNoROICheck.contains(mName);
}

std::vector<ConvertedArgument *> ConvertedFunction::GetInputImages()
{
    std::vector<ConvertedArgument *> ret;
    for (ConvertedArgument &arg : mArguments)
    {
        if (arg.IsInputImage())
        {
            ret.push_back(&arg);
        }
    }
    return ret;
}
std::vector<ConvertedArgument *> ConvertedFunction::GetOutputImages()
{
    std::vector<ConvertedArgument *> ret;
    // if (!IsGetBufferSizeFunction())
    {
        for (ConvertedArgument &arg : mArguments)
        {
            if (arg.IsOutputImage())
            {
                ret.push_back(&arg);
            }
        }
    }
    return ret;
}

int ConvertedFunction::GetInChannels() const
{
    std::string channel = NPPParser::GetChannelString(mFunction.name);

    channel.erase(std::remove(channel.begin(), channel.end(), 'I'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 'M'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 'S'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 'f'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 's'), channel.end());

    if (channel == "C1R" || channel == "AC1R" || channel == "C1" || channel == "C1CR" || channel == "C1C2R" ||
        channel == "C1C3R" || channel == "C1C4R" || channel == "C1AC4R")
    {
        return 1;
    }
    if (channel == "C2R" || channel == "C2" || channel == "C2CR" || channel == "C2C1R" || channel == "C2C3R" ||
        channel == "C2C4R" || channel == "C2P2R" || channel == "C2P3R" || channel == "P2R" || channel == "P2P3R" ||
        channel == "P2C2R" || channel == "P2C3R" || channel == "P2C4R")
    {
        return 2;
    }
    if (channel == "C3R" || channel == "C3" || channel == "C3CR" || channel == "C3C1R" || channel == "C3C2R" ||
        channel == "C3C4R" || channel == "C3A0C4R" || channel == "C3P3R" || channel == "C3P2R" || channel == "P3R" ||
        channel == "P3P2R" || channel == "P3C2R" || channel == "P3C3R" || channel == "P3C4R" || channel == "P3AC4R")
    {
        return 3;
    }
    if (channel == "C4R" || channel == "C4" || channel == "C4CR" || channel == "C4C1R" || channel == "C4C3R" ||
        channel == "C4P4R" || channel == "C4P3R" || channel == "P4R" || channel == "P4P3R" || channel == "P4C3R" ||
        channel == "P4C4R" || channel == "AC4R" || channel == "AC4CR" || channel == "AC4C1R" || channel == "AC4C2R" ||
        channel == "AC4P3R" || channel == "AC4P4R" || channel == "AP4R" || channel == "AP4C4R")
    {
        return 4;
    }
    return 0;
}

int ConvertedFunction::GetOutChannels() const
{
    if (mName == "Compare" || mName == "CompareA" || mName == "CompareEqualEps" || mName == "CompareEqualEpsA")
    {
        return 1;
    }

    std::string channel = NPPParser::GetChannelString(mFunction.name);

    channel.erase(std::remove(channel.begin(), channel.end(), 'I'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 'M'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 'S'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 'f'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 's'), channel.end());

    if (channel == "C1R" || channel == "AC1R" || channel == "C2C1R" || channel == "C1" || channel == "C1CR" ||
        channel == "C3C1R" || channel == "C4C1R" || channel == "P1R" || channel == "AC4C1R")
    {
        return 1;
    }
    if (channel == "C2R" || channel == "C2" || channel == "C2CR" || channel == "C1C2R" || channel == "C2P2R" ||
        channel == "P2R" || channel == "AC4C2R" || channel == "C3P2R" || channel == "P3P2R" || channel == "P3C2R" ||
        channel == "C3C2R" || channel == "P2C2R")
    {
        return 2;
    }
    if (channel == "C3R" || channel == "C3" || channel == "C3CR" || channel == "C1C3R" || channel == "C2C3R" ||
        channel == "C3P3R" || channel == "P3R" || channel == "C4P3R" || channel == "AC4P3R" || channel == "P4P3R" ||
        channel == "P4C3R" || channel == "C4C3R" || channel == "P3C3R" || channel == "C2P3R" || channel == "P2C3R" ||
        channel == "P2P3R")
    {
        return 3;
    }
    if (channel == "C3C4R" || channel == "C3A0C4R" || channel == "P3C4R" || channel == "C4R" || channel == "C4" ||
        channel == "C4CR" || channel == "C2C4R" || channel == "C4P4R" || channel == "P4R" || channel == "P4C4R" ||
        channel == "P2C4R" || channel == "P3AC4R" || channel == "C1C4R" || channel == "C1AC4R" || channel == "AC4R" ||
        channel == "AC4CR" || channel == "AC4P4R" || channel == "AP4R" || channel == "AP4C4R")
    {
        return 4;
    }
    return 0;
}

std::string ConvertedFunction::GetTypeString() const
{
    const std::string type = NPPParser::GetTypeString(mFunction.name);
    std::string type2      = type.substr(0, 2);
    std::string type3      = type.substr(0, 3);
    std::string type4      = type.substr(0, 4);

    if (type2 == "8u" || type2 == "8s")
    {
        return type2;
    }

    if (type4 == "16sc" || type4 == "32sc" || type4 == "32fc")
    {
        return type4;
    }

    if (type3 == "16u" || type3 == "16s" || type3 == "16f" || type3 == "32u" || type3 == "32s" || type3 == "32f" ||
        type3 == "64f")
    {
        return type3;
    }
    return "Unknown";
}

std::string ConvertedFunction::GetImageViewType() const
{
    return std::string("Image") + GetImageTypeShort() + "View";
}

std::string ConvertedFunction::GetImageTypeShort() const
{
    if (mFunction.name == "nppiAddProduct_16f_C1IR_Ctx")
    {
        return "16fC1";
    }
    if (mName == "AddSquare" || mName == "AddProduct" || mName == "AddWeighted")
    {
        return "32fC1";
    }
    const std::string channelCount = std::to_string(GetInChannels());
    const std::string type         = GetTypeString();
    return type + "C" + channelCount;
}

bool ConvertedFunction::IsAlphaIgnored() const
{
    const std::string channel = NPPParser::GetChannelString(mFunction.name);
    // it contains an A in channel string, but is not ignoring alpha...
    if (mName == "AlphaComp" && GetInChannels() == 1)
    {
        return false;
    }
    return channel.find('A') != std::string::npos;
    /*channel.erase(std::remove(channel.begin(), channel.end(), 'I'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 'M'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 'S'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 'f'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 's'), channel.end());

    return channel == "AC4R" || channel == "AC4CR" || channel == "AC4C1R" || channel == "AC4C2R" ||
           channel == "AC4P3R" || channel == "AC4P4R" || channel == "AP4R" || channel == "AP4C4R";*/
}

bool ConvertedFunction::IsInputPlanar() const
{
    return NPPParser::IsPlanarSource(mFunction);
}
bool ConvertedFunction::IsOutputPlanar() const
{
    return NPPParser::IsPlanarDest(mFunction);
}
int ConvertedFunction::InputPlanarCount() const
{
    return NPPParser::GetPlanarSrcCount(mFunction);
}
int ConvertedFunction::OutputPlanarCount() const
{
    return NPPParser::GetPlanarDestCount(mFunction);
}

std::string ConvertedFunction::GetNeededNPPHeaders(const std::vector<ConvertedFunction> &aFunctions)
{
    std::stringstream ss;
    std::set<std::string> categories;

    for (const auto &f : aFunctions)
    {
        categories.insert(f.InnerFunction().category);
    }

    for (const auto &cat : categories)
    {
        ss << sHeaders.at(cat) << std::endl;
    }
    return ss.str();
}
std::string ConvertedFunction::GetNeededImageHeaders(const std::vector<ConvertedFunction> &aFunctions)
{
    std::stringstream ss;
    std::set<std::string> imageTypes;

    for (const auto &f : aFunctions)
    {
        for (const auto &arg : f.Arguments())
        {
            if (arg.Type().find("Image") != std::string::npos && arg.Type().find("View") != std::string::npos)
            {
                std::string imageType = arg.Type();
                if (imageType.substr(0, 5) == "const")
                {
                    // remove "const "
                    imageType = imageType.substr(6);
                }
                if (imageType.find('&') != std::string::npos)
                {
                    imageType = imageType.substr(0, imageType.size() - 2);
                }
                if (imageType.find('*') != std::string::npos)
                {
                    imageType = imageType.substr(0, imageType.size() - 2);
                }
                imageTypes.insert(imageType);
            }
        }
    }

    for (const auto &imageType : imageTypes)
    {
        std::string it = imageType;
        it[0]          = 'i'; // header start with small i...
        ss << "#include \"" << it << ".h\" //NOLINT" << std::endl;
    }
    return ss.str();
}
std::string ConvertedFunction::GetNeededImageForwardDecl(const std::vector<ConvertedFunction> &aFunctions)
{
    std::stringstream ss;
    std::set<std::string> imageTypes;
    std::set<std::string> imageTypesOfFunction;

    for (const auto &f : aFunctions)
    {
        imageTypesOfFunction.insert(f.GetImageViewType());
        for (const auto &arg : f.Arguments())
        {
            if (arg.Type().find("Image") != std::string::npos && arg.Type().find("View") != std::string::npos)
            {
                std::string imageType = arg.Type();
                if (imageType.substr(0, 5) == "const")
                {
                    // remove "const "
                    imageType = imageType.substr(6);
                }
                if (imageType.find('&') != std::string::npos)
                {
                    imageType = imageType.substr(0, imageType.size() - 2);
                }
                if (imageType.find('*') != std::string::npos)
                {
                    imageType = imageType.substr(0, imageType.size() - 2);
                }
                imageTypes.insert(imageType);
            }
        }
    }

    for (const auto &imageType : imageTypes)
    {
        if (!imageTypesOfFunction.contains(imageType))
        {
            ss << "class " << imageType << ";" << std::endl;
        }
    }
    return ss.str();
}
} // namespace mpp::utilities::nppParser
