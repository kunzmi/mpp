// Creates the class members for Npp::ImageView

#include "convertedArgument.h"
#include "convertedFunction.h"
#include "headerTemplates.h"
#include <algorithm>
#include <common/image/pixelTypes.h>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <json.h>
#include <string>
#include <utilities/nppParser/function.h>
#include <utilities/nppParser/nppParser.h>
#include <vector>

using namespace opp::utilities::nppParser;

std::string ReplaceAll(std::string aSrc, const std::string &aToReplace, const std::string &aReplaceBy)
{
    size_t pos = aSrc.find(aToReplace);
    while (pos != std::string::npos)
    {
        aSrc.replace(pos, aToReplace.size(), aReplaceBy);
        pos = aSrc.find(aToReplace, pos + aReplaceBy.size());
    }
    return aSrc;
}

size_t processForPixelType(const std::vector<Function> &aFunctions, std::vector<Function> &aBatchFunction,
                           const std::string &aTypeString, opp::image::PixelTypeEnum aType,
                           const std::string &aCondition, nlohmann::json &aJson,
                           std::vector<Function> &aConversionCheck)
{
    // get all functions for type
    std::vector<Function> functionsForType;
    for (const auto &elem : aFunctions)
    {
        if (elem.category != "support") // we don't need malloc and free here, they are implemented in the image-class
        {
            if (NPPParser::GetPixelType(elem) == aType)
            {
                functionsForType.push_back(elem);
            }
            if (GetChannelCount(aType) == 4)
            {
                opp::image::PixelTypeEnum alphaType = opp::image::PixelTypeEnum::Unknown;
                switch (aType)
                {
                    case opp::image::PixelTypeEnum::PTE64fC4:
                        alphaType = opp::image::PixelTypeEnum::PTE64fC4A;
                        break;
                    case opp::image::PixelTypeEnum::PTE32fC4:
                        alphaType = opp::image::PixelTypeEnum::PTE32fC4A;
                        break;
                    case opp::image::PixelTypeEnum::PTE16fC4:
                        alphaType = opp::image::PixelTypeEnum::PTE16fC4A;
                        break;
                    case opp::image::PixelTypeEnum::PTE16bfC4:
                        alphaType = opp::image::PixelTypeEnum::PTE16bfC4A;
                        break;
                    case opp::image::PixelTypeEnum::PTE32sC4:
                        alphaType = opp::image::PixelTypeEnum::PTE32sC4A;
                        break;
                    case opp::image::PixelTypeEnum::PTE32uC4:
                        alphaType = opp::image::PixelTypeEnum::PTE32uC4A;
                        break;
                    case opp::image::PixelTypeEnum::PTE16sC4:
                        alphaType = opp::image::PixelTypeEnum::PTE16sC4A;
                        break;
                    case opp::image::PixelTypeEnum::PTE16uC4:
                        alphaType = opp::image::PixelTypeEnum::PTE16uC4A;
                        break;
                    case opp::image::PixelTypeEnum::PTE8sC4:
                        alphaType = opp::image::PixelTypeEnum::PTE8sC4A;
                        break;
                    case opp::image::PixelTypeEnum::PTE8uC4:
                        alphaType = opp::image::PixelTypeEnum::PTE8uC4A;
                        break;
                    case opp::image::PixelTypeEnum::PTE16scC4:
                    case opp::image::PixelTypeEnum::PTE32scC4:
                    case opp::image::PixelTypeEnum::PTE32fcC4:
                        continue;
                    default:
                        break;
                }
                if (NPPParser::GetPixelType(elem) == alphaType)
                {
                    functionsForType.push_back(elem);
                }
            }
        }
    }
    std::cout << "Processing " << functionsForType.size() << " functions for " << aTypeString << "." << std::endl;
    std::vector<ConvertedFunction> converted;

    for (auto &func : functionsForType)
    {
        if (func.name.find("Batch") != std::string::npos)
        {
            aBatchFunction.push_back(func);
            continue;
        }
        if (aType == opp::image::PixelTypeEnum::PTE8uC2)
        {
            for (auto &arg : func.arguments)
            {
                if (arg.type == "const Npp8u *const ")
                {
                    arg.type = "const Npp8u *";
                }
            }
        }
        converted.emplace_back(func);
        aConversionCheck.push_back(func);
    }

    const std::string nppHeaders   = ConvertedFunction::GetNeededNPPHeaders(converted);
    const std::string imageHeaders = ConvertedFunction::GetNeededImageHeaders(converted);
    const std::string imageDecl    = ConvertedFunction::GetNeededImageForwardDecl(converted);

    const std::string headerFilename = "image" + aTypeString + "View.h";
    const std::string cppFilename    = "image" + aTypeString + "View.cpp";

    std::ofstream header(std::filesystem::path(DEFAULT_OUT_DIR) / "image" / headerFilename);
    std::ofstream cpp(std::filesystem::path(DEFAULT_OUT_DIR) / "image" / cppFilename);

    const std::string imageTypeShort = converted[0].GetImageTypeShort();
    const std::string toReplace      = "####";
    const std::string header2        = ReplaceAll(headerHeader2, toReplace, imageTypeShort);
    const std::string cpp2           = ReplaceAll(cppHeader2, toReplace, imageTypeShort);

    header << headerHeader1;
    header << imageDecl;
    header << header2;
    header << "#if " << aCondition << std::endl;

    cpp << "// Code automatically generated by nppCodeGenerator - do not modify manually" << std::endl;
    cpp << "#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)" << std::endl;
    cpp << "#if OPP_ENABLE_NPP_BACKEND" << std::endl;
    cpp << "#include <common/image/pixelTypeEnabler.h> //NOLINT(misc-include-cleaner)" << std::endl;
    cpp << std::endl;
    cpp << imageHeaders;
    cpp << cppHeader1;
    cpp << nppHeaders;
    cpp << cpp2;
    cpp << "#if " << aCondition << std::endl << std::endl;

    for (auto &conv : converted)
    {
        header << conv.ToStringHeader() << std::endl;
        cpp << conv.ToStringCpp() << std::endl;
    }

    header << std::endl << "#endif // " << aCondition << std::endl;
    cpp << "#endif // " << aCondition << std::endl;
    header << headerFooter;
    cpp << cppFooter;

    aJson[aTypeString] = converted;
    return converted.size();
}

int main()
{
    try
    {
        std::vector<Function> functions = NPPParser::GetFunctions();

        std::vector<Function> functionsNoCtxDoublets;

        for (auto &elem : functions)
        {
            if (NPPParser::IsCtx(elem))
            {
                functionsNoCtxDoublets.push_back(elem);
            }
            else
            {
                const std::string ctxName = elem.name + "_Ctx";

                auto iter = std::find_if(functions.begin(), functions.end(),
                                         [&ctxName](const Function &aFunc) { return aFunc.name == ctxName; });
                if (iter == functions.end())
                {
                    // function has no _Ctx counterpart so we have to keep it
                    functionsNoCtxDoublets.push_back(elem);
                }
            }
        }

        /*for (const auto &f : functions)
        {
            if (f.name == "nppiMinEvery_8u_C4IR_Ctx")
            {
                ConvertedFunction cf(f);
            }
        }*/

        std::vector<Function> batchFunc;
        std::vector<Function> convertedFuncCheck;
        size_t totalConverted = 0;

        nlohmann::json aj;

        // 8u
        {
            // 8uC1
            {
                const std::string condition          = "OPPi_ENABLE_UINT8_TYPE && OPPi_ENABLE_ONE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE8uC1;
                const std::string typeString         = "8uC1";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 8uC2
            {
                const std::string condition          = "OPPi_ENABLE_UINT8_TYPE && OPPi_ENABLE_TWO_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE8uC2;
                const std::string typeString         = "8uC2";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 8uC3
            {
                const std::string condition          = "OPPi_ENABLE_UINT8_TYPE && OPPi_ENABLE_THREE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE8uC3;
                const std::string typeString         = "8uC3";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 8uC4
            {
                const std::string condition          = "OPPi_ENABLE_UINT8_TYPE && OPPi_ENABLE_FOUR_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE8uC4;
                const std::string typeString         = "8uC4";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }
        }

        // 8s
        {
            // 8sC1
            {
                const std::string condition          = "OPPi_ENABLE_INT8_TYPE && OPPi_ENABLE_ONE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE8sC1;
                const std::string typeString         = "8sC1";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 8sC2
            {
                const std::string condition          = "OPPi_ENABLE_INT8_TYPE && OPPi_ENABLE_TWO_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE8sC2;
                const std::string typeString         = "8sC2";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 8sC3
            {
                const std::string condition          = "OPPi_ENABLE_INT8_TYPE && OPPi_ENABLE_THREE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE8sC3;
                const std::string typeString         = "8sC3";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 8sC4
            {
                const std::string condition          = "OPPi_ENABLE_INT8_TYPE && OPPi_ENABLE_FOUR_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE8sC4;
                const std::string typeString         = "8sC4";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }
        }

        // 16u
        {
            // 16uC1
            {
                const std::string condition          = "OPPi_ENABLE_UINT16_TYPE && OPPi_ENABLE_ONE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE16uC1;
                const std::string typeString         = "16uC1";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 16uC2
            {
                const std::string condition          = "OPPi_ENABLE_UINT16_TYPE && OPPi_ENABLE_TWO_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE16uC2;
                const std::string typeString         = "16uC2";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 16uC3
            {
                const std::string condition          = "OPPi_ENABLE_UINT16_TYPE && OPPi_ENABLE_THREE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE16uC3;
                const std::string typeString         = "16uC3";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 16uC4
            {
                const std::string condition          = "OPPi_ENABLE_UINT16_TYPE && OPPi_ENABLE_FOUR_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE16uC4;
                const std::string typeString         = "16uC4";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }
        }

        // 16s
        {
            // 16sC1
            {
                const std::string condition          = "OPPi_ENABLE_SINT16_TYPE && OPPi_ENABLE_ONE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE16sC1;
                const std::string typeString         = "16sC1";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 16sC2
            {
                const std::string condition          = "OPPi_ENABLE_SINT16_TYPE && OPPi_ENABLE_TWO_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE16sC2;
                const std::string typeString         = "16sC2";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 16sC3
            {
                const std::string condition          = "OPPi_ENABLE_SINT16_TYPE && OPPi_ENABLE_THREE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE16sC3;
                const std::string typeString         = "16sC3";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 16sC4
            {
                const std::string condition          = "OPPi_ENABLE_SINT16_TYPE && OPPi_ENABLE_FOUR_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE16sC4;
                const std::string typeString         = "16sC4";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }
        }

        // 16sc
        {
            // 16scC1
            {
                const std::string condition          = "OPPi_ENABLE_COMPLEX_INT16_TYPE && OPPi_ENABLE_ONE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE16scC1;
                const std::string typeString         = "16scC1";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 16scC2
            {
                const std::string condition          = "OPPi_ENABLE_COMPLEX_INT16_TYPE && OPPi_ENABLE_TWO_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE16scC2;
                const std::string typeString         = "16scC2";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 16scC3
            {
                const std::string condition          = "OPPi_ENABLE_COMPLEX_INT16_TYPE && OPPi_ENABLE_THREE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE16scC3;
                const std::string typeString         = "16scC3";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 16scC4
            {
                const std::string condition          = "OPPi_ENABLE_COMPLEX_INT16_TYPE && OPPi_ENABLE_FOUR_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE16scC4;
                const std::string typeString         = "16scC4";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }
        }

        // 32u
        {
            // 32uC1
            {
                const std::string condition          = "OPPi_ENABLE_UINT32_TYPE && OPPi_ENABLE_ONE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE32uC1;
                const std::string typeString         = "32uC1";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 32uC2
            {
                const std::string condition          = "OPPi_ENABLE_UINT32_TYPE && OPPi_ENABLE_TWO_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE32uC2;
                const std::string typeString         = "32uC2";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 32uC3
            {
                const std::string condition          = "OPPi_ENABLE_UINT32_TYPE && OPPi_ENABLE_THREE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE32uC3;
                const std::string typeString         = "32uC3";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 32uC4
            {
                const std::string condition          = "OPPi_ENABLE_UINT32_TYPE && OPPi_ENABLE_FOUR_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE32uC4;
                const std::string typeString         = "32uC4";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }
        }

        // 32s
        {
            // 32sC1
            {
                const std::string condition          = "OPPi_ENABLE_INT32_TYPE && OPPi_ENABLE_ONE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE32sC1;
                const std::string typeString         = "32sC1";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 32sC2
            {
                const std::string condition          = "OPPi_ENABLE_INT32_TYPE && OPPi_ENABLE_TWO_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE32sC2;
                const std::string typeString         = "32sC2";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 32sC3
            {
                const std::string condition          = "OPPi_ENABLE_INT32_TYPE && OPPi_ENABLE_THREE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE32sC3;
                const std::string typeString         = "32sC3";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 32sC4
            {
                const std::string condition          = "OPPi_ENABLE_INT32_TYPE && OPPi_ENABLE_FOUR_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE32sC4;
                const std::string typeString         = "32sC4";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }
        }

        // 32sc
        {
            // 32scC1
            {
                const std::string condition          = "OPPi_ENABLE_COMPLEX_INT32_TYPE && OPPi_ENABLE_ONE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE32scC1;
                const std::string typeString         = "32scC1";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 32scC2
            {
                const std::string condition          = "OPPi_ENABLE_COMPLEX_INT32_TYPE && OPPi_ENABLE_TWO_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE32scC2;
                const std::string typeString         = "32scC2";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 32scC3
            {
                const std::string condition          = "OPPi_ENABLE_COMPLEX_INT32_TYPE && OPPi_ENABLE_THREE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE32scC3;
                const std::string typeString         = "32scC3";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 32scC4
            {
                const std::string condition          = "OPPi_ENABLE_COMPLEX_INT32_TYPE && OPPi_ENABLE_FOUR_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE32scC4;
                const std::string typeString         = "32scC4";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }
        }

        // 32f
        {
            // 32fC1
            {
                const std::string condition          = "OPPi_ENABLE_FLOAT_TYPE && OPPi_ENABLE_ONE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE32fC1;
                const std::string typeString         = "32fC1";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 32fC2
            {
                const std::string condition          = "OPPi_ENABLE_FLOAT_TYPE && OPPi_ENABLE_TWO_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE32fC2;
                const std::string typeString         = "32fC2";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 32fC3
            {
                const std::string condition          = "OPPi_ENABLE_FLOAT_TYPE && OPPi_ENABLE_THREE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE32fC3;
                const std::string typeString         = "32fC3";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 32fC4
            {
                const std::string condition          = "OPPi_ENABLE_FLOAT_TYPE && OPPi_ENABLE_FOUR_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE32fC4;
                const std::string typeString         = "32fC4";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }
        }

        // 32fc
        {
            // 32fcC1
            {
                const std::string condition          = "OPPi_ENABLE_COMPLEX_FLOAT_TYPE && OPPi_ENABLE_ONE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE32fcC1;
                const std::string typeString         = "32fcC1";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 32fcC2
            {
                const std::string condition          = "OPPi_ENABLE_COMPLEX_FLOAT_TYPE && OPPi_ENABLE_TWO_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE32fcC2;
                const std::string typeString         = "32fcC2";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 32fcC3
            {
                const std::string condition          = "OPPi_ENABLE_COMPLEX_FLOAT_TYPE && OPPi_ENABLE_THREE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE32fcC3;
                const std::string typeString         = "32fcC3";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 32fcC4
            {
                const std::string condition          = "OPPi_ENABLE_COMPLEX_FLOAT_TYPE && OPPi_ENABLE_FOUR_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE32fcC4;
                const std::string typeString         = "32fcC4";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }
        }

        // 64f
        {
            // 64fC1
            {
                const std::string condition          = "OPPi_ENABLE_DOUBLE_TYPE && OPPi_ENABLE_ONE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE64fC1;
                const std::string typeString         = "64fC1";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 64fC2
            {
                const std::string condition          = "OPPi_ENABLE_DOUBLE_TYPE && OPPi_ENABLE_TWO_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE64fC2;
                const std::string typeString         = "64fC2";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 64fC3
            {
                const std::string condition          = "OPPi_ENABLE_DOUBLE_TYPE && OPPi_ENABLE_THREE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE64fC3;
                const std::string typeString         = "64fC3";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 64fC4
            {
                const std::string condition          = "OPPi_ENABLE_DOUBLE_TYPE && OPPi_ENABLE_FOUR_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE64fC4;
                const std::string typeString         = "64fC4";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }
        }

        // 16f
        {
            // 16fC1
            {
                const std::string condition          = "OPPi_ENABLE_HALFFLOAT16_TYPE && OPPi_ENABLE_ONE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE16fC1;
                const std::string typeString         = "16fC1";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 16fC2
            {
                const std::string condition          = "OPPi_ENABLE_HALFFLOAT16_TYPE && OPPi_ENABLE_TWO_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE16fC2;
                const std::string typeString         = "16fC2";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 16fC3
            {
                const std::string condition          = "OPPi_ENABLE_HALFFLOAT16_TYPE && OPPi_ENABLE_THREE_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE16fC3;
                const std::string typeString         = "16fC3";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }

            // 16fC4
            {
                const std::string condition          = "OPPi_ENABLE_HALFFLOAT16_TYPE && OPPi_ENABLE_FOUR_CHANNEL";
                const opp::image::PixelTypeEnum type = opp::image::PixelTypeEnum::PTE16fC4;
                const std::string typeString         = "16fC4";

                totalConverted += processForPixelType(functionsNoCtxDoublets, batchFunc, typeString, type, condition,
                                                      aj, convertedFuncCheck);
            }
        }

        {
            std::ofstream file(std::filesystem::path(DEFAULT_OUT_DIR) / "../../../nppFunctions.json");

            if (file.fail())
            {
                std::cout << "Failed to write JSON file with NPP function definitions.";
            }

            file << std::setw(4);
            file << aj;
        }

        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << "Not converted batch functions:" << std::endl;

        for (const auto &elem : batchFunc)
        {
            std::cout << elem.name << std::endl;
        }

        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << "Unknown pixel type:" << std::endl;

        // get all functions for unknown pixel type
        std::vector<Function> unknown;
        for (auto &elem : functionsNoCtxDoublets)
        {
            if (elem.category != "support") // we don't need malloc and free here
            {
                if (NPPParser::GetPixelType(elem) == opp::image::PixelTypeEnum::Unknown)
                {
                    unknown.push_back(elem);
                    std::cout << elem.name << std::endl;
                }
            }
        }

        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << "Skipped support functions:" << std::endl;
        // get all functions for support (just to count them)
        std::vector<Function> support;
        for (auto &elem : functionsNoCtxDoublets)
        {
            if (elem.category == "support")
            {
                support.push_back(elem);
                std::cout << elem.name << std::endl;
            }
        }

        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << "Didn't convert " << batchFunc.size() << " functions for batch processing." << std::endl;
        std::cout << "Unknown pixel type: " << unknown.size() << " functions." << std::endl;
        std::cout << "Support functions: " << support.size() << std::endl;
        std::cout << "Converted " << totalConverted << " functions." << std::endl;

        std::cout << "Total number of functions: " << functionsNoCtxDoublets.size() << std::endl;
    }
    catch (...)
    {
        return 1;
    }
    return 0;
}
