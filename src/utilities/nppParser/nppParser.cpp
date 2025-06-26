#include "function.h"
#include "nppParser.h"
#include <algorithm>
#include <clang-c/CXSourceLocation.h>
#include <clang-c/CXString.h>
#include <clang-c/Index.h>
#include <common/image/pixelTypes.h>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace mpp::utilities::nppParser
{
const std::filesystem::path NPPParser::_basePath = NPP_HEADER_DIR; // NOLINT(cert-err58-cpp)

const std::vector<std::string> NPPParser::_headerFileList = std::vector<std::string>( // NOLINT(cert-err58-cpp)
    {"nppi_support_functions.h", "nppi_data_exchange_and_initialization.h", "nppi_arithmetic_and_logical_operations.h",
     "nppi_color_conversion.h", "nppi_threshold_and_compare_operations.h", "nppi_morphological_operations.h",
     "nppi_filtering_functions.h", "nppi_statistics_functions.h", "nppi_linear_transforms.h",
     "nppi_geometry_transforms.h"});

const std::vector<std::string> NPPParser::_categoryList = std::vector<std::string>( // NOLINT(cert-err58-cpp)
    {"support", "data exchange and initialization", "arithmetic and logical", "color conversion",
     "threshold and compare", "morphological", "filtering", "statistics", "linear transforms", "geometry transforms"});

std::vector<Function> NPPParser::GetFunctions()
{
    std::vector<Function> ret;

    for (std::size_t i = 0; i < _headerFileList.size(); i++)
    {
        std::vector<Function> functions = GetFunctions(i);

        ret.insert(ret.end(), functions.begin(), functions.end());
    }

    return ret;
}

std::vector<Function> NPPParser::GetFunctions(std::size_t aHeaderID)
{
    if (aHeaderID >= _headerFileList.size())
    {
        return {};
    }
    std::vector<Function> ret;

    const std::filesystem::path filename = GetFileName(aHeaderID);

    CXIndex index          = clang_createIndex(0, 0); // Create index
    CXTranslationUnit unit = clang_parseTranslationUnit(index, filename.generic_string().c_str(), nullptr, 0, nullptr,
                                                        0, CXTranslationUnit_None);

    if (unit == nullptr)
    {
        return {}; // failed to open file
    }

    const CXCursor cursor = clang_getTranslationUnitCursor(unit); // Obtain a cursor at the root of the translation unit
    clang_visitChildren(cursor, CursorVisitor, &ret);

    for (auto &elem : ret)
    {
        elem.category = _categoryList[aHeaderID];
    }

    return ret;
}

std::filesystem::path NPPParser::GetFileName(std::size_t aHeaderID)
{
    if (aHeaderID >= _headerFileList.size())
    {
        return {};
    }
    std::filesystem::path filename = _basePath / _headerFileList[aHeaderID];
    return filename;
}

std::string NPPParser::Convert(const CXString &aString)
{
    std::string result = clang_getCString(aString);
    clang_disposeString(aString);
    return result;
}

Function NPPParser::GetPrototype(CXCursor aCursor)
{
    const CXType type = clang_getCursorType(aCursor);
    Function f;
    f.name       = Convert(clang_getCursorSpelling(aCursor));
    f.returnType = Convert(clang_getTypeSpelling(clang_getResultType(type)));

    const int num_args = clang_Cursor_getNumArguments(aCursor);
    for (int i = 0; i < num_args; ++i)
    {
        Argument a;
        const CXCursor arg_cursor = clang_Cursor_getArgument(aCursor, std::uint32_t(i));
        a.name                    = Convert(clang_getCursorSpelling(arg_cursor));
        a.type                    = Convert(clang_getTypeSpelling(clang_getArgType(type, std::uint32_t(i))));
        f.arguments.emplace_back(a);
    }
    return f;
}

CXChildVisitResult NPPParser::CursorVisitor(CXCursor aCursor, CXCursor /*aParent*/, CXClientData aClientData)
{
    std::vector<Function> *functions = static_cast<std::vector<Function> *>(aClientData);
    if (functions == nullptr)
    {
        return CXChildVisit_Break;
    }

    if (clang_Location_isFromMainFile(clang_getCursorLocation(aCursor)) == 0)
    {
        return CXChildVisit_Continue;
    }

    const CXCursorKind kind = clang_getCursorKind(aCursor);
    if (kind == CXCursorKind::CXCursor_FunctionDecl)
    {
        const Function f = GetPrototype(aCursor);
        functions->emplace_back(f);
    }

    return CXChildVisit_Continue;
}

bool NPPParser::BaseNameHas_(const std::string &aFName)
{
    const size_t first_  = aFName.find('_');
    const size_t second_ = aFName.find('_', first_ + 1);

    std::string shouldBeType = aFName.substr(first_ + 1, second_ - first_ - 1);
    // if the type string doesn't start with a number, it is likely that the actual name contains a '_'

    if (shouldBeType.empty())
    {
        return false;
    }
    if (shouldBeType[0] == '8' || shouldBeType[0] == '1' || shouldBeType[0] == '3' || shouldBeType[0] == '6')
    {
        return false;
    }
    return true;
}

bool NPPParser::IsMergedColorTwist(const std::string &aFName)
{
    return aFName.find("_ColorTwist32f_") < aFName.size();
}

std::string NPPParser::GetTypeString(const std::string &aFName)
{
    if (aFName == "nppiYCCKToCMYK_JPEG_601_8u_P4R" || aFName == "nppiYCCKToCMYK_JPEG_601_8u_P4R_Ctx")
    {
        return "8u";
    }

    if (!BaseNameHas_(aFName))
    {
        const size_t first_  = aFName.find('_');
        const size_t second_ = aFName.find('_', first_ + 1);
        return aFName.substr(first_ + 1, second_ - first_ - 1);
    }
    const size_t first_  = aFName.find('_');
    const size_t second_ = aFName.find('_', first_ + 1);
    const size_t third_  = aFName.find('_', second_ + 1);
    return aFName.substr(second_ + 1, third_ - second_ - 1);
}

std::string NPPParser::GetChannelString(const std::string &aFName)
{
    if (aFName == "nppiYCCKToCMYK_JPEG_601_8u_P4R" || aFName == "nppiYCCKToCMYK_JPEG_601_8u_P4R_Ctx")
    {
        return "P4R";
    }

    if (!BaseNameHas_(aFName) && !IsMergedColorTwist(aFName))
    {
        const size_t first_  = aFName.find('_');
        const size_t second_ = aFName.find('_', first_ + 1);
        const size_t third_  = aFName.find('_', second_ + 1);
        return aFName.substr(second_ + 1, third_ - second_ - 1);
    }

    const size_t first_  = aFName.find('_');
    const size_t second_ = aFName.find('_', first_ + 1);
    const size_t third_  = aFName.find('_', second_ + 1);
    const size_t fourth_ = aFName.find('_', third_ + 1);
    return aFName.substr(third_ + 1, fourth_ - third_ - 1);
}

std::string NPPParser::GetBaseName(const std::string &aFName)
{
    if (aFName == "nppiYCCKToCMYK_JPEG_601_8u_P4R" || aFName == "nppiYCCKToCMYK_JPEG_601_8u_P4R_Ctx")
    {
        return "nppiYCCKToCMYK_JPEG_601";
    }
    if (aFName == "nppiFusedAbsDiff_Threshold_GTVal_Ctx" || aFName == "nppiFusedAbsDiff_Threshold_GTVal_I_Ctx")
    {
        return "nppiFusedAbsDiff_Threshold_GTVal";
    }

    if (BaseNameHas_(aFName))
    {
        const size_t first_  = aFName.find('_');
        const size_t second_ = aFName.find('_', first_ + 1);
        return aFName.substr(0, second_);
    }

    return aFName.substr(0, aFName.find('_'));
}

std::string nppParser::NPPParser::GetShortName(const Function &aFunction)
{
    std::string baseName = GetBaseName(aFunction.name);

    const size_t pos = baseName.find("DeviceC");
    if (pos != std::string::npos)
    {
        baseName.replace(pos, 7, "");
    }
    if (baseName[baseName.size() - 1] == 'C')
    {
        return baseName.substr(4, baseName.size() - 1 - 4);
    }
    if (baseName == "nppiMulCScale")
    {
        return "MulScale";
    }

    if (aFunction.name == "nppiSumGetBufferHostSize_8u64s_C1R_Ctx")
    {
        return "SumGetBufferHostSize64";
    }
    if (aFunction.name == "nppiValidNormLevelGetBufferHostSize_8u32f_C1R_Ctx")
    {
        return "ValidNormLevelGetBufferHostSize32f";
    }
    if (aFunction.name == "nppiSameNormLevelGetBufferHostSize_8u32f_C1R_Ctx")
    {
        return "SameNormLevelGetBufferHostSize32f";
    }
    if (aFunction.name == "nppiFullNormLevelGetBufferHostSize_8u32f_C1R_Ctx")
    {
        return "FullNormLevelGetBufferHostSize32f";
    }

    return baseName.substr(4);
}

bool NPPParser::GetContext(const std::string &aFName)
{
    return aFName.substr(aFName.size() - 3) == "Ctx";
}
mpp::image::PixelTypeEnum nppParser::NPPParser::GetPixelType(const Function &aFunction)
{
    std::string channel    = GetChannelString(aFunction.name);
    const std::string type = GetTypeString(aFunction.name);

    const std::string shortName = GetShortName(aFunction);

    if (aFunction.name == "nppiAddProduct_16f_C1IR_Ctx")
    {
        // special case...
        return mpp::image::PixelTypeEnum::PTE16fC1;
    }

    if (shortName == "AddSquare" || shortName == "AddProduct" || shortName == "AddWeighted")
    {
        // special case...
        return mpp::image::PixelTypeEnum::PTE32fC1;
    }
    if (aFunction.name == "nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx" ||
        aFunction.name == "nppiContoursImageMarchingSquaresInterpolation_32f_C1R_Ctx" ||
        aFunction.name == "nppiContoursImageMarchingSquaresInterpolation_64f_C1R_Ctx")
    {
        // better call directly the NPP function for these ones...
        return mpp::image::PixelTypeEnum::Unknown;
    }

    channel.erase(std::remove(channel.begin(), channel.end(), 'I'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 'M'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 'S'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 'f'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 's'), channel.end());

    const std::string type2 = type.substr(0, 2);
    const std::string type3 = type.substr(0, 3);
    const std::string type4 = type.substr(0, 4);

    if (type2 == "8u")
    {
        if (channel == "C1R" || channel == "AC1R" || channel == "C1" || channel == "C1CR" || channel == "C1C2R" ||
            channel == "C1C3R" || channel == "C1C4R" || channel == "C1AC4R" || channel == "P1R")
        {
            return mpp::image::PixelTypeEnum::PTE8uC1;
        }
        if (channel == "C2R" || channel == "C2" || channel == "C2CR" || channel == "C2C1R" || channel == "C2C3R" ||
            channel == "C2C4R" || channel == "C2P2R" || channel == "C2P3R" || channel == "P2R" || channel == "P2P3R" ||
            channel == "P2C2R" || channel == "P2C3R" || channel == "P2C4R")
        {
            return mpp::image::PixelTypeEnum::PTE8uC2;
        }
        if (channel == "C3R" || channel == "C3" || channel == "C3CR" || channel == "C3C1R" || channel == "C3C2R" ||
            channel == "C3C4R" || channel == "C3A0C4R" || channel == "C3P3R" || channel == "C3P2R" ||
            channel == "P3R" || channel == "P3P2R" || channel == "P3C2R" || channel == "P3C3R" || channel == "P3C4R" ||
            channel == "P3AC4R")
        {
            return mpp::image::PixelTypeEnum::PTE8uC3;
        }
        if (channel == "C4R" || channel == "C4" || channel == "C4CR" || channel == "C4C1R" || channel == "C4C3R" ||
            channel == "C4P4R" || channel == "C4P3R" || channel == "P4R" || channel == "P4P3R" || channel == "P4C3R" ||
            channel == "P4C4R")
        {
            return mpp::image::PixelTypeEnum::PTE8uC4;
        }
        if (channel == "AC4R" || channel == "AC4CR" || channel == "AC4C1R" || channel == "AC4C2R" ||
            channel == "AC4P3R" || channel == "AC4P4R" || channel == "AP4R" || channel == "AP4C4R")
        {
            return mpp::image::PixelTypeEnum::PTE8uC4A;
        }
    }

    if (type2 == "8s")
    {
        if (channel == "C1R" || channel == "AC1R" || channel == "C1" || channel == "C1CR" || channel == "C1C2R" ||
            channel == "C1C3R" || channel == "C1C4R" || channel == "C1AC4R" || channel == "P1R")
        {
            return mpp::image::PixelTypeEnum::PTE8sC1;
        }
        if (channel == "C2R" || channel == "C2" || channel == "C2CR" || channel == "C2C1R" || channel == "C2C3R" ||
            channel == "C2C4R" || channel == "C2P2R" || channel == "C2P3R" || channel == "P2R" || channel == "P2P3R" ||
            channel == "P2C2R" || channel == "P2C3R" || channel == "P2C4R")
        {
            return mpp::image::PixelTypeEnum::PTE8sC2;
        }
        if (channel == "C3R" || channel == "C3" || channel == "C3CR" || channel == "C3C1R" || channel == "C3C2R" ||
            channel == "C3C4R" || channel == "C3A0C4R" || channel == "C3P3R" || channel == "C3P2R" ||
            channel == "P3R" || channel == "P3P2R" || channel == "P3C2R" || channel == "P3C3R" || channel == "P3C4R" ||
            channel == "P3AC4R")
        {
            return mpp::image::PixelTypeEnum::PTE8sC3;
        }
        if (channel == "C4R" || channel == "C4" || channel == "C4CR" || channel == "C4C1R" || channel == "C4C3R" ||
            channel == "C4P4R" || channel == "C4P3R" || channel == "P4R" || channel == "P4P3R" || channel == "P4C3R" ||
            channel == "P4C4R")
        {
            return mpp::image::PixelTypeEnum::PTE8sC4;
        }
        if (channel == "AC4R" || channel == "AC4CR" || channel == "AC4C1R" || channel == "AC4C2R" ||
            channel == "AC4P3R" || channel == "AC4P4R" || channel == "AP4R" || channel == "AP4C4R")
        {
            return mpp::image::PixelTypeEnum::PTE8sC4A;
        }
    }

    if (type3 == "16f")
    {
        if (channel == "C1R" || channel == "AC1R" || channel == "C1" || channel == "C1CR" || channel == "C1C2R" ||
            channel == "C1C3R" || channel == "C1C4R" || channel == "C1AC4R" || channel == "P1R")
        {
            return mpp::image::PixelTypeEnum::PTE16fC1;
        }
        if (channel == "C2R" || channel == "C2" || channel == "C2CR" || channel == "C2C1R" || channel == "C2C3R" ||
            channel == "C2C4R" || channel == "C2P2R" || channel == "C2P3R" || channel == "P2R" || channel == "P2P3R" ||
            channel == "P2C2R" || channel == "P2C3R" || channel == "P2C4R")
        {
            return mpp::image::PixelTypeEnum::PTE16fC2;
        }
        if (channel == "C3R" || channel == "C3" || channel == "C3CR" || channel == "C3C1R" || channel == "C3C2R" ||
            channel == "C3C4R" || channel == "C3A0C4R" || channel == "C3P3R" || channel == "C3P2R" ||
            channel == "P3R" || channel == "P3P2R" || channel == "P3C2R" || channel == "P3C3R" || channel == "P3C4R" ||
            channel == "P3AC4R")
        {
            return mpp::image::PixelTypeEnum::PTE16fC3;
        }
        if (channel == "C4R" || channel == "C4" || channel == "C4CR" || channel == "C4C1R" || channel == "C4C3R" ||
            channel == "C4P4R" || channel == "C4P3R" || channel == "P4R" || channel == "P4P3R" || channel == "P4C3R" ||
            channel == "P4C4R")
        {
            return mpp::image::PixelTypeEnum::PTE16fC4;
        }
        if (channel == "AC4R" || channel == "AC4CR" || channel == "AC4C1R" || channel == "AC4C2R" ||
            channel == "AC4P3R" || channel == "AC4P4R" || channel == "AP4R" || channel == "AP4C4R")
        {
            return mpp::image::PixelTypeEnum::PTE16fC4A;
        }
    }

    if (type3 == "16u")
    {
        if (channel == "C1R" || channel == "AC1R" || channel == "C1" || channel == "C1CR" || channel == "C1C2R" ||
            channel == "C1C3R" || channel == "C1C4R" || channel == "C1AC4R" || channel == "P1R")
        {
            return mpp::image::PixelTypeEnum::PTE16uC1;
        }
        if (channel == "C2R" || channel == "C2" || channel == "C2CR" || channel == "C2C1R" || channel == "C2C3R" ||
            channel == "C2C4R" || channel == "C2P2R" || channel == "C2P3R" || channel == "P2R" || channel == "P2P3R" ||
            channel == "P2C2R" || channel == "P2C3R" || channel == "P2C4R")
        {
            return mpp::image::PixelTypeEnum::PTE16uC2;
        }
        if (channel == "C3R" || channel == "C3" || channel == "C3CR" || channel == "C3C1R" || channel == "C3C2R" ||
            channel == "C3C4R" || channel == "C3A0C4R" || channel == "C3P3R" || channel == "C3P2R" ||
            channel == "P3R" || channel == "P3P2R" || channel == "P3C2R" || channel == "P3C3R" || channel == "P3C4R" ||
            channel == "P3AC4R")
        {
            return mpp::image::PixelTypeEnum::PTE16uC3;
        }
        if (channel == "C4R" || channel == "C4" || channel == "C4CR" || channel == "C4C1R" || channel == "C4C3R" ||
            channel == "C4P4R" || channel == "C4P3R" || channel == "P4R" || channel == "P4P3R" || channel == "P4C3R" ||
            channel == "P4C4R")
        {
            return mpp::image::PixelTypeEnum::PTE16uC4;
        }
        if (channel == "AC4R" || channel == "AC4CR" || channel == "AC4C1R" || channel == "AC4C2R" ||
            channel == "AC4P3R" || channel == "AC4P4R" || channel == "AP4R" || channel == "AP4C4R")
        {
            return mpp::image::PixelTypeEnum::PTE16uC4A;
        }
    }

    if (type4 == "16sc")
    {
        if (channel == "C1R" || channel == "AC1R" || channel == "C1" || channel == "C1CR" || channel == "C1C2R" ||
            channel == "C1C3R" || channel == "C1C4R" || channel == "C1AC4R" || channel == "P1R")
        {
            return mpp::image::PixelTypeEnum::PTE16scC1;
        }
        if (channel == "C2R" || channel == "C2" || channel == "C2CR" || channel == "C2C1R" || channel == "C2C3R" ||
            channel == "C2C4R" || channel == "C2P2R" || channel == "C2P3R" || channel == "P2R" || channel == "P2P3R" ||
            channel == "P2C2R" || channel == "P2C3R" || channel == "P2C4R")
        {
            return mpp::image::PixelTypeEnum::PTE16scC2;
        }
        if (channel == "C3R" || channel == "C3" || channel == "C3CR" || channel == "C3C1R" || channel == "C3C2R" ||
            channel == "C3C4R" || channel == "C3A0C4R" || channel == "C3P3R" || channel == "C3P2R" ||
            channel == "P3R" || channel == "P3P2R" || channel == "P3C2R" || channel == "P3C3R" || channel == "P3C4R" ||
            channel == "P3AC4R")
        {
            return mpp::image::PixelTypeEnum::PTE16scC3;
        }
        if (channel == "C4R" || channel == "C4" || channel == "C4CR" || channel == "C4C1R" || channel == "C4C3R" ||
            channel == "C4P4R" || channel == "C4P3R" || channel == "P4R" || channel == "P4P3R" || channel == "P4C3R" ||
            channel == "P4C4R")
        {
            return mpp::image::PixelTypeEnum::PTE16scC4;
        }
        if (channel == "AC4R" || channel == "AC4CR" || channel == "AC4C1R" || channel == "AC4C2R" ||
            channel == "AC4P3R" || channel == "AC4P4R" || channel == "AP4R" || channel == "AP4C4R")
        {
            return mpp::image::PixelTypeEnum::PTE16scC4;
        } /**/
    }

    if (type3 == "16s")
    {
        if (channel == "C1R" || channel == "AC1R" || channel == "C1" || channel == "C1CR" || channel == "C1C2R" ||
            channel == "C1C3R" || channel == "C1C4R" || channel == "C1AC4R" || channel == "P1R")
        {
            return mpp::image::PixelTypeEnum::PTE16sC1;
        }
        if (channel == "C2R" || channel == "C2" || channel == "C2CR" || channel == "C2C1R" || channel == "C2C3R" ||
            channel == "C2C4R" || channel == "C2P2R" || channel == "C2P3R" || channel == "P2R" || channel == "P2P3R" ||
            channel == "P2C2R" || channel == "P2C3R" || channel == "P2C4R")
        {
            return mpp::image::PixelTypeEnum::PTE16sC2;
        }
        if (channel == "C3R" || channel == "C3" || channel == "C3CR" || channel == "C3C1R" || channel == "C3C2R" ||
            channel == "C3C4R" || channel == "C3A0C4R" || channel == "C3P3R" || channel == "C3P2R" ||
            channel == "P3R" || channel == "P3P2R" || channel == "P3C2R" || channel == "P3C3R" || channel == "P3C4R" ||
            channel == "P3AC4R")
        {
            return mpp::image::PixelTypeEnum::PTE16sC3;
        }
        if (channel == "C4R" || channel == "C4" || channel == "C4CR" || channel == "C4C1R" || channel == "C4C3R" ||
            channel == "C4P4R" || channel == "C4P3R" || channel == "P4R" || channel == "P4P3R" || channel == "P4C3R" ||
            channel == "P4C4R")
        {
            return mpp::image::PixelTypeEnum::PTE16sC4;
        }
        if (channel == "AC4R" || channel == "AC4CR" || channel == "AC4C1R" || channel == "AC4C2R" ||
            channel == "AC4P3R" || channel == "AC4P4R" || channel == "AP4R" || channel == "AP4C4R")
        {
            return mpp::image::PixelTypeEnum::PTE16sC4A;
        }
    }

    if (type3 == "32u")
    {
        if (channel == "C1R" || channel == "AC1R" || channel == "C1" || channel == "C1CR" || channel == "C1C2R" ||
            channel == "C1C3R" || channel == "C1C4R" || channel == "C1AC4R" || channel == "P1R")
        {
            return mpp::image::PixelTypeEnum::PTE32uC1;
        }
        if (channel == "C2R" || channel == "C2" || channel == "C2CR" || channel == "C2C1R" || channel == "C2C3R" ||
            channel == "C2C4R" || channel == "C2P2R" || channel == "C2P3R" || channel == "P2R" || channel == "P2P3R" ||
            channel == "P2C2R" || channel == "P2C3R" || channel == "P2C4R")
        {
            return mpp::image::PixelTypeEnum::PTE32uC2;
        }
        if (channel == "C3R" || channel == "C3" || channel == "C3CR" || channel == "C3C1R" || channel == "C3C2R" ||
            channel == "C3C4R" || channel == "C3A0C4R" || channel == "C3P3R" || channel == "C3P2R" ||
            channel == "P3R" || channel == "P3P2R" || channel == "P3C2R" || channel == "P3C3R" || channel == "P3C4R" ||
            channel == "P3AC4R")
        {
            return mpp::image::PixelTypeEnum::PTE32uC3;
        }
        if (channel == "C4R" || channel == "C4" || channel == "C4CR" || channel == "C4C1R" || channel == "C4C3R" ||
            channel == "C4P4R" || channel == "C4P3R")
        {
            return mpp::image::PixelTypeEnum::PTE32uC4;
        }
        if (channel == "AC4R" || channel == "AC4CR" || channel == "AC4C1R" || channel == "AC4C2R" ||
            channel == "AC4P3R" || channel == "AC4P4R" || channel == "AP4R" || channel == "AP4C4R")
        {
            return mpp::image::PixelTypeEnum::PTE32uC4A;
        }
    }

    if (type4 == "32sc")
    {
        if (channel == "C1R" || channel == "AC1R" || channel == "C1" || channel == "C1CR" || channel == "C1C2R" ||
            channel == "C1C3R" || channel == "C1C4R" || channel == "C1AC4R" || channel == "P1R")
        {
            return mpp::image::PixelTypeEnum::PTE32scC1;
        }
        if (channel == "C2R" || channel == "C2" || channel == "C2CR" || channel == "C2C1R" || channel == "C2C3R" ||
            channel == "C2C4R" || channel == "C2P2R" || channel == "C2P3R" || channel == "P2R" || channel == "P2P3R" ||
            channel == "P2C2R" || channel == "P2C3R" || channel == "P2C4R")
        {
            return mpp::image::PixelTypeEnum::PTE32scC2;
        }
        if (channel == "C3R" || channel == "C3" || channel == "C3CR" || channel == "C3C1R" || channel == "C3C2R" ||
            channel == "C3C4R" || channel == "C3A0C4R" || channel == "C3P3R" || channel == "C3P2R" ||
            channel == "P3R" || channel == "P3P2R" || channel == "P3C2R" || channel == "P3C3R" || channel == "P3C4R" ||
            channel == "P3AC4R")
        {
            return mpp::image::PixelTypeEnum::PTE32scC3;
        }
        if (channel == "C4R" || channel == "C4" || channel == "C4CR" || channel == "C4C1R" || channel == "C4C3R" ||
            channel == "C4P4R" || channel == "C4P3R" || channel == "P4R" || channel == "P4P3R" || channel == "P4C3R" ||
            channel == "P4C4R")
        {
            return mpp::image::PixelTypeEnum::PTE32scC4;
        }
        if (channel == "AC4R" || channel == "AC4CR" || channel == "AC4C1R" || channel == "AC4C2R" ||
            channel == "AC4P3R" || channel == "AC4P4R" || channel == "AP4R" || channel == "AP4C4R")
        {
            return mpp::image::PixelTypeEnum::PTE32scC4;
        } /* */
    }

    if (type3 == "32s")
    {
        if (channel == "C1R" || channel == "AC1R" || channel == "C1" || channel == "C1CR" || channel == "C1C2R" ||
            channel == "C1C3R" || channel == "C1C4R" || channel == "C1AC4R" || channel == "P1R")
        {
            return mpp::image::PixelTypeEnum::PTE32sC1;
        }
        if (channel == "C2R" || channel == "C2" || channel == "C2CR" || channel == "C2C1R" || channel == "C2C3R" ||
            channel == "C2C4R" || channel == "C2P2R" || channel == "C2P3R" || channel == "P2R" || channel == "P2P3R" ||
            channel == "P2C2R" || channel == "P2C3R" || channel == "P2C4R")
        {
            return mpp::image::PixelTypeEnum::PTE32sC2;
        }
        if (channel == "C3R" || channel == "C3" || channel == "C3CR" || channel == "C3C1R" || channel == "C3C2R" ||
            channel == "C3C4R" || channel == "C3A0C4R" || channel == "C3P3R" || channel == "C3P2R" ||
            channel == "P3R" || channel == "P3P2R" || channel == "P3C2R" || channel == "P3C3R" || channel == "P3C4R" ||
            channel == "P3AC4R")
        {
            return mpp::image::PixelTypeEnum::PTE32sC3;
        }
        if (channel == "C4R" || channel == "C4" || channel == "C4CR" || channel == "C4C1R" || channel == "C4C3R" ||
            channel == "C4P4R" || channel == "C4P3R" || channel == "P4R" || channel == "P4P3R" || channel == "P4C3R" ||
            channel == "P4C4R")
        {
            return mpp::image::PixelTypeEnum::PTE32sC4;
        }
        if (channel == "AC4R" || channel == "AC4CR" || channel == "AC4C1R" || channel == "AC4C2R" ||
            channel == "AC4P3R" || channel == "AC4P4R" || channel == "AP4R" || channel == "AP4C4R")
        {
            return mpp::image::PixelTypeEnum::PTE32sC4A;
        }
    }

    if (type4 == "32fc")
    {
        if (channel == "C1R" || channel == "AC1R" || channel == "C1" || channel == "C1CR" || channel == "C1C2R" ||
            channel == "C1C3R" || channel == "C1C4R" || channel == "C1AC4R" || channel == "P1R")
        {
            return mpp::image::PixelTypeEnum::PTE32fcC1;
        }
        if (channel == "C2R" || channel == "C2" || channel == "C2CR" || channel == "C2C1R" || channel == "C2C3R" ||
            channel == "C2C4R" || channel == "C2P2R" || channel == "C2P3R" || channel == "P2R" || channel == "P2P3R" ||
            channel == "P2C2R" || channel == "P2C3R" || channel == "P2C4R")
        {
            return mpp::image::PixelTypeEnum::PTE32fcC2;
        }
        if (channel == "C3R" || channel == "C3" || channel == "C3CR" || channel == "C3C1R" || channel == "C3C2R" ||
            channel == "C3C4R" || channel == "C3A0C4R" || channel == "C3P3R" || channel == "C3P2R" ||
            channel == "P3R" || channel == "P3P2R" || channel == "P3C2R" || channel == "P3C3R" || channel == "P3C4R" ||
            channel == "P3AC4R")
        {
            return mpp::image::PixelTypeEnum::PTE32fcC3;
        }
        if (channel == "C4R" || channel == "C4" || channel == "C4CR" || channel == "C4C1R" || channel == "C4C3R" ||
            channel == "C4P4R" || channel == "C4P3R" || channel == "P4R" || channel == "P4P3R" || channel == "P4C3R" ||
            channel == "P4C4R")
        {
            return mpp::image::PixelTypeEnum::PTE32fcC4;
        }
        if (channel == "AC4R" || channel == "AC4CR" || channel == "AC4C1R" || channel == "AC4C2R" ||
            channel == "AC4P3R" || channel == "AC4P4R" || channel == "AP4R" || channel == "AP4C4R")
        {
            return mpp::image::PixelTypeEnum::PTE32fcC4;
        } /* */
    }

    if (type3 == "32f")
    {
        if (channel == "C1R" || channel == "AC1R" || channel == "C1" || channel == "C1CR" || channel == "C1C2R" ||
            channel == "C1C3R" || channel == "C1C4R" || channel == "C1AC4R" || channel == "P1R")
        {
            return mpp::image::PixelTypeEnum::PTE32fC1;
        }
        if (channel == "C2R" || channel == "C2" || channel == "C2CR" || channel == "C2C1R" || channel == "C2C3R" ||
            channel == "C2C4R" || channel == "C2P2R" || channel == "C2P3R" || channel == "P2R" || channel == "P2P3R" ||
            channel == "P2C2R" || channel == "P2C3R" || channel == "P2C4R")
        {
            return mpp::image::PixelTypeEnum::PTE32fC2;
        }
        if (channel == "C3R" || channel == "C3" || channel == "C3CR" || channel == "C3C1R" || channel == "C3C2R" ||
            channel == "C3C4R" || channel == "C3A0C4R" || channel == "C3P3R" || channel == "C3P2R" ||
            channel == "P3R" || channel == "P3P2R" || channel == "P3C2R" || channel == "P3C3R" || channel == "P3C4R" ||
            channel == "P3AC4R")
        {
            return mpp::image::PixelTypeEnum::PTE32fC3;
        }
        if (channel == "C4R" || channel == "C4" || channel == "C4CR" || channel == "C4C1R" || channel == "C4C3R" ||
            channel == "C4P4R" || channel == "C4P3R" || channel == "P4R" || channel == "P4P3R" || channel == "P4C3R" ||
            channel == "P4C4R")
        {
            return mpp::image::PixelTypeEnum::PTE32fC4;
        }
        if (channel == "AC4R" || channel == "AC4CR" || channel == "AC4C1R" || channel == "AC4C2R" ||
            channel == "AC4P3R" || channel == "AC4P4R" || channel == "AP4R" || channel == "AP4C4R")
        {
            return mpp::image::PixelTypeEnum::PTE32fC4A;
        }
    }

    if (type3 == "64f")
    {
        if (channel == "C1R" || channel == "AC1R" || channel == "C1" || channel == "C1CR" || channel == "C1C2R" ||
            channel == "C1C3R" || channel == "C1C4R" || channel == "C1AC4R" || channel == "P1R")
        {
            return mpp::image::PixelTypeEnum::PTE64fC1;
        }
        if (channel == "C2R" || channel == "C2" || channel == "C2CR" || channel == "C2C1R" || channel == "C2C3R" ||
            channel == "C2C4R" || channel == "C2P2R" || channel == "C2P3R" || channel == "P2R" || channel == "P2P3R" ||
            channel == "P2C2R" || channel == "P2C3R" || channel == "P2C4R")
        {
            return mpp::image::PixelTypeEnum::PTE64fC2;
        }
        if (channel == "C3R" || channel == "C3" || channel == "C3CR" || channel == "C3C1R" || channel == "C3C2R" ||
            channel == "C3C4R" || channel == "C3A0C4R" || channel == "C3P3R" || channel == "C3P2R" ||
            channel == "P3R" || channel == "P3P2R" || channel == "P3C2R" || channel == "P3C3R" || channel == "P3C4R" ||
            channel == "P3AC4R")
        {
            return mpp::image::PixelTypeEnum::PTE64fC3;
        }
        if (channel == "C4R" || channel == "C4" || channel == "C4CR" || channel == "C4C1R" || channel == "C4C3R" ||
            channel == "C4P4R" || channel == "C4P3R" || channel == "P4R" || channel == "P4P3R" || channel == "P4C3R" ||
            channel == "P4C4R")
        {
            return mpp::image::PixelTypeEnum::PTE64fC4;
        }
        if (channel == "AC4R" || channel == "AC4CR" || channel == "AC4C1R" || channel == "AC4C2R" ||
            channel == "AC4P3R" || channel == "AC4P4R" || channel == "AP4R" || channel == "AP4C4R")
        {
            return mpp::image::PixelTypeEnum::PTE64fC4A;
        }
    }
    return mpp::image::PixelTypeEnum::Unknown;
}
bool nppParser::NPPParser::IsPlanar(const Function &aFunction)
{
    const std::string channel = GetChannelString(aFunction.name);
    return channel.find('P') < channel.size();
}
bool nppParser::NPPParser::IsPlanarSource(const Function &aFunction)
{
    return GetPlanarSrcCount(aFunction) > 0;
}
bool nppParser::NPPParser::IsPlanarDest(const Function &aFunction)
{
    return GetPlanarDestCount(aFunction) > 0;
}
int nppParser::NPPParser::GetPlanarDestCount(const Function &aFunction)
{
    std::string channel = GetChannelString(aFunction.name);

    channel.erase(std::remove(channel.begin(), channel.end(), 'I'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 'M'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 'S'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 'f'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 's'), channel.end());

    if (channel == "C3P2R" || channel == "C2P2R" || channel == "P2R" || channel == "P3P2R")
    {
        return 2;
    }
    if (channel == "C4P3R" || channel == "P4P3R" || channel == "AC4P3R" || channel == "C3P3R" || channel == "P3R" ||
        channel == "P2P3R" || channel == "C2P3R")
    {
        return 3;
    }
    if (channel == "C4P4R" || channel == "P4R" || channel == "AC4P4R" || channel == "AP4R")
    {
        return 4;
    }
    return 0;
}
int nppParser::NPPParser::GetPlanarSrcCount(const Function &aFunction)
{
    std::string channel = GetChannelString(aFunction.name);

    channel.erase(std::remove(channel.begin(), channel.end(), 'I'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 'M'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 'S'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 'f'), channel.end());
    channel.erase(std::remove(channel.begin(), channel.end(), 's'), channel.end());

    if (channel == "P4R" || channel == "P4P3R" || channel == "P4C3R" || channel == "P4C4R" || channel == "AP4R" ||
        channel == "AP4C4R")
    {
        return 4;
    }
    if (channel == "P3R" || channel == "P3P2R" || channel == "P3C2R" || channel == "P3C3R" || channel == "P3C4R" ||
        channel == "P3AC4R")
    {
        return 3;
    }
    if (channel == "P2R" || channel == "P2P3R" || channel == "P2C2R" || channel == "P2C3R" || channel == "P2C4R")
    {
        return 2;
    }
    return 0;
}
bool nppParser::NPPParser::IsInplace(const Function &aFunction)
{
    const std::string baseName = GetShortName(aFunction);
    if (baseName == "Set")
    {
        return true;
    }
    const std::string channel = GetChannelString(aFunction.name);
    return channel.find('I') < channel.size();
}
bool nppParser::NPPParser::IsMasked(const Function &aFunction)
{
    const std::string channel = GetChannelString(aFunction.name);
    return channel.find('M') < channel.size();
}
bool nppParser::NPPParser::IsSfs(const Function &aFunction)
{
    const std::string channel = GetChannelString(aFunction.name);
    return channel.find("Sfs") < channel.size();
}
bool nppParser::NPPParser::IsCtx(const Function &aFunction)
{
    return GetContext(aFunction.name);
}
bool nppParser::NPPParser::IsConstant(const Function &aFunction)
{
    const std::string baseName = GetBaseName(aFunction.name);

    if (baseName[baseName.size() - 1] == 'C')
    {
        return true;
    }

    if (baseName == "nppiMulCScale")
    {
        return true;
    }

    return false;
}
bool nppParser::NPPParser::IsDeviceConstant(const Function &aFunction)
{
    const std::string baseName = GetBaseName(aFunction.name);

    return baseName.find("DeviceC") != std::string::npos;
}
} // namespace mpp::utilities::nppParser