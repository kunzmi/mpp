#pragma once

#include "function.h"
#include <clang-c/Index.h>
#include <common/image/pixelTypes.h>
#include <filesystem>
#include <json.h>
#include <string>
#include <vector>

namespace opp::utilities::nppParser
{

class NPPParser
{
  public:
    NPPParser()  = default;
    ~NPPParser() = default;

    static std::vector<Function> GetFunctions();

    static bool BaseNameHas_(const std::string &aFName);

    static bool IsMergedColorTwist(const std::string &aFName);

    static std::string GetTypeString(const std::string &aFName);

    static std::string GetChannelString(const std::string &aFName);

    static std::string GetBaseName(const std::string &aFName);

    static std::string GetShortName(Function &aFunction);

    static bool GetContext(const std::string &aFName);

    static opp::image::PixelTypeEnum GetPixelType(Function &aFunction);

    static bool IsPlanar(Function &aFunction);

    static bool IsInplace(Function &aFunction);

    static bool IsMasked(Function &aFunction);

    static bool IsSfs(Function &aFunction);

    static bool IsCtx(Function &aFunction);

    static bool IsConstant(Function &aFunction);

    static bool IsDeviceConstant(Function &aFunction);

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

} // namespace opp::utilities::nppParser
