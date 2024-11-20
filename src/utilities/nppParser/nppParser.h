#pragma once

#include "function.h"
#include <clang-c/Index.h>
#include <filesystem>
#include <json.h>
#include <string>
#include <vector>

namespace nppParser
{
class NPPParser
{
  public:
    NPPParser()  = default;
    ~NPPParser() = default;

    static std::vector<Function> GetFunctions();

  private:
    static const std::filesystem::path _basePath;
    static const std::vector<std::string> _headerFileList;
    static const std::vector<std::string> _categoryList;

    static std::filesystem::path GetFileName(size_t aHeaderID);

    static std::string Convert(const CXString &aString);

    static Function GetPrototype(CXCursor aCursor);

    static CXChildVisitResult CursorVisitor(CXCursor aCursor, CXCursor aParent, CXClientData aClientData);

    static std::vector<Function> GetFunctions(size_t aHeaderID);
};

} // namespace nppParser
