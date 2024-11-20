#include "function.h"
#include "nppParser.h"
#include <clang-c/CXSourceLocation.h>
#include <clang-c/CXString.h>
#include <clang-c/Index.h>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace nppParser
{
const std::filesystem::path NPPParser::_basePath = // NOLINT(cert-err58-cpp)
    R"(C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include\)";

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

} // namespace nppParser