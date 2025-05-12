#pragma once
#include "convertedArgument.h"
#include <json.h>
#include <map>
#include <string>
#include <unordered_set>
#include <utilities/nppParser/function.h>
#include <vector>

namespace opp::utilities::nppParser
{
class ConvertedFunction
{
  private:
    // These functions have a missing const specifier for the source image
    inline static const std::unordered_set<std::string> sMissingConst = { // NOLINT
        "LabelMarkersUF", "DistanceTransformPBA", "DistanceTransformAbsPBA", "SignedDistanceTransformPBA",
        "SignedDistanceTransformAbsPBA"};

    // These functions take the allocated pointer as input
    inline static const std::unordered_set<std::string> sGeometryWithAllocPointer = { // NOLINT
        "Remap",
        "Resize",
        "ResizeSqrPixel",
        "Rotate",
        "WarpAffine",
        "WarpAffineBack",
        "WarpAffineQuad",
        "WarpPerspective",
        "WarpPerspectiveBack",
        "WarpPerspectiveQuad"};

    // These functions take the allocated pointer as input
    inline static const std::unordered_set<std::string> sCFAToRGBWithAllocPointer = { // NOLINT
        "CFAToRGB", "CFAToRGBA"};

    // These functions need an additional channel argument
    inline static const std::unordered_set<std::string> sCopyInsert = { // NOLINT
        "nppiCopy_8u_C1C3R_Ctx",  "nppiCopy_8u_C1C4R_Ctx",  "nppiCopy_16s_C1C3R_Ctx", "nppiCopy_16s_C1C4R_Ctx",
        "nppiCopy_16u_C1C3R_Ctx", "nppiCopy_16u_C1C4R_Ctx", "nppiCopy_32s_C1C3R_Ctx", "nppiCopy_32s_C1C4R_Ctx",
        "nppiCopy_32f_C1C2R_Ctx", "nppiCopy_32f_C1C3R_Ctx", "nppiCopy_32f_C1C4R_Ctx"};

    // These functions need an additional channel argument
    inline static const std::unordered_set<std::string> sCopyExtract = { // NOLINT
        "nppiCopy_8u_C3C1R_Ctx",  "nppiCopy_8u_C4C1R_Ctx",  "nppiCopy_16s_C3C1R_Ctx", "nppiCopy_16s_C4C1R_Ctx",
        "nppiCopy_16u_C3C1R_Ctx", "nppiCopy_16u_C4C1R_Ctx", "nppiCopy_32s_C3C1R_Ctx", "nppiCopy_32s_C4C1R_Ctx",
        "nppiCopy_32f_C2C1R_Ctx", "nppiCopy_32f_C3C1R_Ctx", "nppiCopy_32f_C4C1R_Ctx"};

    // These functions need an additional channel argument for input and output
    inline static const std::unordered_set<std::string> sCopyChannel = { // NOLINT
        "nppiCopy_8u_C3CR_Ctx",  "nppiCopy_8u_C4CR_Ctx",  "nppiCopy_16s_C3CR_Ctx", "nppiCopy_16s_C4CR_Ctx",
        "nppiCopy_16u_C3CR_Ctx", "nppiCopy_16u_C4CR_Ctx", "nppiCopy_32s_C3CR_Ctx", "nppiCopy_32s_C4CR_Ctx",
        "nppiCopy_32f_C3CR_Ctx", "nppiCopy_32f_C4CR_Ctx"};

    // These functions need an additional channel argument output
    inline static const std::unordered_set<std::string> sSetChannel = { // NOLINT
        "nppiSet_8u_C3CR_Ctx",  "nppiSet_8u_C4CR_Ctx",  "nppiSet_16s_C3CR_Ctx", "nppiSet_16s_C4CR_Ctx",
        "nppiSet_16u_C3CR_Ctx", "nppiSet_16u_C4CR_Ctx", "nppiSet_32s_C3CR_Ctx", "nppiSet_32s_C4CR_Ctx",
        "nppiSet_32f_C3CR_Ctx", "nppiSet_32f_C4CR_Ctx"};

    // These functions need an additional "Float" in theie name to have unique names
    inline static const std::unordered_set<std::string> sAddFloatToName = { // NOLINT
        "nppiFullNormLevelGetBufferHostSize_8u32f_C3R_Ctx",  "nppiSameNormLevelGetBufferHostSize_8u32f_C3R_Ctx",
        "nppiValidNormLevelGetBufferHostSize_8u32f_C3R_Ctx", "nppiFullNormLevelGetBufferHostSize_8u32f_C4R_Ctx",
        "nppiFullNormLevelGetBufferHostSize_8u32f_AC4R_Ctx", "nppiSameNormLevelGetBufferHostSize_8u32f_C4R_Ctx",
        "nppiSameNormLevelGetBufferHostSize_8u32f_AC4R_Ctx", "nppiValidNormLevelGetBufferHostSize_8u32f_C4R_Ctx",
        "nppiValidNormLevelGetBufferHostSize_8u32f_AC4R_Ctx"};

    inline static const std::map<std::string, std::string> sHeaders =
        std::map<std::string, std::string>( // NOLINT(cert-err58-cpp)
            {{"support", "#include <nppi_support_functions.h> //NOLINT"},
             {"data exchange and initialization", "#include <nppi_data_exchange_and_initialization.h> //NOLINT"},
             {"arithmetic and logical", "#include <nppi_arithmetic_and_logical_operations.h> //NOLINT"},
             {"color conversion", "#include <nppi_color_conversion.h> //NOLINT"},
             {"threshold and compare", "#include <nppi_threshold_and_compare_operations.h> //NOLINT"},
             {"morphological", "#include <nppi_morphological_operations.h> //NOLINT"},
             {"filtering", "#include <nppi_filtering_functions.h> //NOLINT"},
             {"statistics", "#include <nppi_statistics_functions.h> //NOLINT"},
             {"linear transforms", "#include <nppi_linear_transforms.h> //NOLINT"},
             {"geometry transforms", "#include <nppi_geometry_transforms.h> //NOLINT"}});

  public:
    explicit ConvertedFunction(const Function &aFunction);
    ~ConvertedFunction() = default;

    ConvertedFunction(const ConvertedFunction &)     = default;
    ConvertedFunction(ConvertedFunction &&) noexcept = default;

    ConvertedFunction &operator=(const ConvertedFunction &)     = default;
    ConvertedFunction &operator=(ConvertedFunction &&) noexcept = default;

    std::string ToStringHeader() const;
    std::string ToStringCpp() const;

    std::vector<ConvertedArgument *> GetInputImages();
    std::vector<ConvertedArgument *> GetOutputImages();

    const std::vector<ConvertedArgument> &Arguments() const
    {
        return mArguments;
    }
    std::vector<ConvertedArgument> &Arguments()
    {
        return mArguments;
    }

    const Function &InnerFunction() const
    {
        return mFunction;
    }

    const std::string &Name() const
    {
        return mName;
    }
    std::string &Name()
    {
        return mName;
    }

    bool IsStatic() const
    {
        return mIsStatic;
    }
    bool IsConst() const
    {
        return mIsConst;
    }
    bool IsMasked() const
    {
        return mIsMasked;
    }

    bool &IsStatic()
    {
        return mIsStatic;
    }
    bool &IsConst()
    {
        return mIsConst;
    }
    bool &IsMasked()
    {
        return mIsMasked;
    }

    bool IsMissingConst() const;
    bool IsInplace() const;
    bool IsGetBufferSizeFunction() const;
    bool IsFullROIFunction() const;
    bool IsGeometryFunction() const;
    bool IsCFAToRGBFunction() const;

    bool IsCopyInsert() const;
    bool IsCopyExtract() const;
    bool IsCopyChannel() const;
    bool IsSetChannel() const;

    int GetInChannels() const;
    int GetOutChannels() const;
    std::string GetTypeString() const;
    std::string GetImageViewType() const;
    std::string GetImageTypeShort() const;

    bool IsAlphaIgnored() const;

    bool IsInputPlanar() const;
    bool IsOutputPlanar() const;
    int InputPlanarCount() const;
    int OutputPlanarCount() const;

    static std::string GetNeededNPPHeaders(const std::vector<ConvertedFunction> &aFunctions);
    static std::string GetNeededImageHeaders(const std::vector<ConvertedFunction> &aFunctions);
    static std::string GetNeededImageForwardDecl(const std::vector<ConvertedFunction> &aFunctions);

    friend void to_json(nlohmann::json &aj, const ConvertedFunction &aFunction);

  private:
    const Function &mFunction;
    std::string mReturnType;
    std::string mName;
    std::string mCallHeader;
    std::string mCallFooter;
    std::string mExceptionMessage;
    std::string mImageViewType;
    std::vector<ConvertedArgument> mArguments;
    bool mIsStatic;
    bool mIsConst;
    bool mIsMasked;
};

inline void to_json(nlohmann::json &aj, const ConvertedFunction &aFunction)
{
    aj = nlohmann::json{{"returnType", aFunction.mReturnType},
                        {"name", aFunction.mName},
                        {"arguments", aFunction.mArguments},
                        {"header", aFunction.ToStringHeader()},
                        {"original", aFunction.mFunction}};
}

} // namespace opp::utilities::nppParser