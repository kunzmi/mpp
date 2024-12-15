
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utilities/nppParser/function.h>
#include <utilities/nppParser/nppParser.h>
#include <vector>

using namespace opp::utilities::nppParser;

std::string GetInType(const Argument &aArgument)
{
    if (aArgument.type == "NppStreamContext")
    {
        return "const NppStreamContext &";
    }
    if (aArgument.name == "pSrc1" || aArgument.name == "pSrc2" || aArgument.name == "pSrc")
    {
        return "const Image8uC1 &";
    }
    if (aArgument.name == "pMask")
    {
        return "const Image8uC1 &";
    }
    if (aArgument.name == "nConstant" && aArgument.type.find('*') == std::string::npos)
    {
        return "const Pixel8uC1 &";
    }
    if (aArgument.name == "pConstant" && aArgument.type.find('*') != std::string::npos)
    {
        if (aArgument.type.find("const") == std::string::npos)
        {
            return "cuda::DevVarView<Pixel8uC1> &";
        }
        return "const cuda::DevVarView<Pixel8uC1> &";
    }
    if (aArgument.type == "Npp8u")
    {
        return "const Pixel8uC1 &";
    }

    return aArgument.type + " ";
}

bool GetIsStep(const Argument &aArgument)
{
    if (aArgument.name.size() > 5 && aArgument.name.substr(aArgument.name.size() - 4) == "Step")
    {
        return true;
    }
    return false;
}

bool SkipInArgument(const Argument &aArgument)
{
    if (aArgument.name == "oSizeROI")
    {
        return true;
    }
    if (aArgument.name == "pSrcDst")
    {
        return true;
    }
    if (aArgument.name == "pDst")
    {
        return true;
    }
    if (aArgument.name == "oSrcSizeROI")
    {
        return true;
    }
    if (aArgument.name == "oDstSizeROI")
    {
        return true;
    }

    return GetIsStep(aArgument);
}

bool SkipOutArgument(const Argument &aArgument)
{
    if (aArgument.name.size() > 5 && aArgument.name.substr(aArgument.name.size() - 4) == "Step")
    {
        return true;
    }
    return false;
}

bool GetIsInputImage(const Argument &aArgument)
{
    if (aArgument.name == "pSrc" || aArgument.name == "pSrc1" || aArgument.name == "pSrc2")
    {
        return true;
    }
    return false;
}

bool GetIsMask(const Argument &aArgument)
{
    if (aArgument.name == "pMask")
    {
        return true;
    }
    return false;
}

bool GetIsOutputImage(const Argument &aArgument)
{
    if (aArgument.name == "pSrcDst" || aArgument.name == "pDst")
    {
        return true;
    }
    return false;
}

std::vector<Argument> GetInputImages(const Function &aFunction)
{
    std::vector<Argument> ret;
    for (const auto &arg : aFunction.arguments)
    {
        if (GetIsInputImage(arg))
        {
            ret.push_back(arg);
        }
    }
    return ret;
}

std::string headerHeader = R"(#pragma once

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


)";
std::string headerFooter = R"(
};
} // namespace opp::image::npp
)";

std::string cppHeader = R"(#include "image8uC1.h"
#include "image8uC1View.h"
#include <backends/cuda/devVarView.h>
#include <backends/npp/nppException.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/size2D.h>
#include <common/safeCast.h>
#include <memory>
#include <nppdefs.h>
#include <nppi_arithmetic_and_logical_operations.h>
#include <nppi_color_conversion.h>
#include <nppi_data_exchange_and_initialization.h>
#include <nppi_filtering_functions.h>
#include <nppi_geometry_transforms.h>
#include <nppi_linear_transforms.h>
#include <nppi_morphological_operations.h>
#include <nppi_statistics_functions.h>
#include <nppi_support_functions.h>
#include <nppi_threshold_and_compare_operations.h>

using namespace opp::cuda;

namespace opp::image::npp
{

Image8uC1::Image8uC1(int aWidth, int aHeight) : Image8uC1(Size2D(aWidth, aHeight))
{
}
Image8uC1::Image8uC1(const Size2D &aSize) : Image8uC1View(aSize)
{
    int pitch    = 0;
    PointerRef() = reinterpret_cast<Pixel8uC1 *>(nppiMalloc_8u_C1(aSize.x, aSize.y, &pitch));
    if (Pointer() == nullptr)
    {
        nppSafeCallExt(NPP_ERROR, "Could not allocate an Image8uC1 image with size " << aSize);
    }
    PitchRef()      = to_size_t(pitch);
    PointerRoiRef() = Pointer();
}
Image8uC1::~Image8uC1()
{
    if (PointerRef() != nullptr)
    {
        nppiFree(Pointer());
    }
    PointerRef()    = nullptr;
    PointerRoiRef() = nullptr;
    PitchRef()      = 0;
    ROIRef()        = Roi();
    SizeAllocRef()  = Size2D();
}

Image8uC1::Image8uC1(Image8uC1 &&aOther) noexcept
{
    PointerRef()    = aOther.PointerRef();
    PointerRoiRef() = aOther.PointerRoiRef();
    PitchRef()      = aOther.PitchRef();
    ROIRef()        = aOther.ROIRef();
    SizeAllocRef()  = aOther.SizeAllocRef();

    aOther.PointerRef()    = nullptr;
    aOther.PointerRoiRef() = nullptr;
    aOther.PitchRef()      = 0;
    aOther.ROIRef()        = Roi();
    aOther.SizeAllocRef()  = Size2D();
}

Image8uC1 &Image8uC1::operator=(Image8uC1 &&aOther) noexcept
{
    if (std::addressof(aOther) == std::addressof(*this))
    {
        return *this;
    }
    PointerRef()    = aOther.PointerRef();
    PointerRoiRef() = aOther.PointerRoiRef();
    PitchRef()      = aOther.PitchRef();
    ROIRef()        = aOther.ROIRef();
    SizeAllocRef()  = aOther.SizeAllocRef();

    aOther.PointerRef()    = nullptr;
    aOther.PointerRoiRef() = nullptr;
    aOther.PitchRef()      = 0;
    aOther.ROIRef()        = Roi();
    aOther.SizeAllocRef()  = Size2D();

    return *this;
}
)";
std::string cppFooter = R"(

} // namespace opp::image::npp
)";

void process8uC1(std::vector<Function> &aFunctions, std::vector<Function> &aFailedFunctions)
{
    std::ofstream header(std::filesystem::path(DEFAULT_OUT_DIR) / "image" / "image8uC1.h");
    std::ofstream cpp(std::filesystem::path(DEFAULT_OUT_DIR) / "image" / "image8uC1.cpp");

    header << headerHeader;
    cpp << cppHeader; /**/

    // get all functions for 8uC1
    std::vector<Function> func8uC1;
    for (auto &elem : aFunctions)
    {
        /*
        {"support", "data exchange and initialization", "arithmetic and logical", "color conversion",
     "threshold and compare", "morphological", "filtering", "statistics", "linear transforms", "geometry transforms"});
        */
        if (elem.category == "arithmetic and logical") // to begin with
        {
            if (NPPParser::GetPixelType(elem) == opp::image::PixelTypeEnum::PTE8uC1)
            {
                func8uC1.push_back(elem);
            }
        }
        if (elem.category == "data exchange and initialization") // to begin with
        {
            if (NPPParser::GetPixelType(elem) == opp::image::PixelTypeEnum::PTE8uC1)
            {
                func8uC1.push_back(elem);
            }
        }
        if (elem.category == "color conversion") // to begin with
        {
            if (NPPParser::GetPixelType(elem) == opp::image::PixelTypeEnum::PTE8uC1)
            {
                func8uC1.push_back(elem);
            }
        }
    }

    // std::sort(func8uC1.begin(), func8uC1.end(),
    //           [](const Function &aA, const Function &aB) { return aA.name < aB.name; });

    // create wrappers:
    for (auto &elem : func8uC1)
    {
        const std::string baseName = NPPParser::GetShortName(elem);

        const bool isInplace = NPPParser::IsInplace(elem);
        const bool isPlanar  = NPPParser::IsPlanar(elem);
        /*const bool isCtx            = NPPParser::IsCtx(elem);
        const bool isMasked         = NPPParser::IsMasked(elem);
        const bool isSfs            = NPPParser::IsSfs(elem);
        const bool isConstant       = NPPParser::IsConstant(elem);
        const bool isDeviceConstant = NPPParser::IsDeviceConstant(elem);*/

        if (isPlanar)
        {
            aFailedFunctions.push_back(elem);
            continue;
        }

        bool ok = true;

        std::stringstream signature;

        std::string returnType = "void";
        if (elem.returnType != "NppStatus")
        {
            returnType = elem.returnType;
        }

        signature << baseName << "(";

        for (const auto &argument : elem.arguments)
        {
            if (!SkipInArgument(argument))
            {
                signature << GetInType(argument) << argument.name << ", ";
            }
        }

        std::string strSig = signature.str();
        strSig             = strSig.substr(0, strSig.size() - 2) + ")";

        std::vector<Argument> inputImages = GetInputImages(elem);

        header << "    " << returnType << " " << strSig << ";" << std::endl;

        cpp << returnType << " Image8uC1::" << strSig << std::endl;
        cpp << "{" << std::endl;

        for (const auto &inputImage : inputImages)
        {
            cpp << "    checkSameSize(ROI(), " << inputImage.name << ".ROI());" << std::endl;
        }

        cpp << "    nppSafeCallExt(" << elem.name << "(";

        for (const auto &argument : elem.arguments)
        {
            if (SkipOutArgument(argument))
            {
                continue;
            }
            else if (GetIsInputImage(argument))
            {
                cpp << "reinterpret_cast<const Npp8u *>(" << argument.name << ".PointerRoi()), to_int(" << argument.name
                    << ".Pitch())";
            }
            else if (GetIsMask(argument))
            {
                cpp << "reinterpret_cast<const Npp8u *>(" << argument.name << ".PointerRoi()), to_int(" << argument.name
                    << ".Pitch())";
            }
            else if (GetIsOutputImage(argument))
            {
                cpp << "reinterpret_cast<Npp8u *>(PointerRoi()), to_int(Pitch())";
            }
            else if (argument.name == "oSizeROI")
            {
                cpp << "NppiSizeRoi()";
            }
            else if (argument.name == "oSrcSizeROI")
            {
                cpp << "pSrc.NppiSizeRoi()";
            }
            else if (argument.name == "oDstSizeROI")
            {
                cpp << "NppiSizeRoi()";
            }
            else if (GetInType(argument) == "const Pixel8uC1 &")
            {
                cpp << argument.name << ".x";
            }
            else if (GetInType(argument).find("DevVarView") != std::string::npos)
            {
                cpp << "reinterpret_cast<" << argument.type << ">(" << argument.name << ".Pointer())";
            }
            else
            {
                cpp << argument.name;
            }

            if (argument.name != elem.arguments[elem.arguments.size() - 1].name)
            {
                cpp << ", ";
            }
        }

        cpp << R"(), )" << std::endl << "                   ";

        if (inputImages.size() == 1)
        {
            cpp << "\"ROI Src: \" << " << inputImages[0].name << ".ROI() << ";
        }
        if (inputImages.size() == 2)
        {
            cpp << "\"ROI Src1: \" << " << inputImages[0].name << ".ROI() << \" ROI Src2: \" << " << inputImages[1].name
                << ".ROI() << ";
        }

        if (inputImages.size() == 0)
        {
            if (isInplace)
            {
                cpp << "\"ROI SrcDst: \" << ROI() ";
            }
            else
            {
                cpp << "\"ROI Dst: \" << ROI() ";
            }
        }
        else
        {
            if (isInplace)
            {
                cpp << "\" ROI SrcDst: \" << ROI() ";
            }
            else
            {
                cpp << "\" ROI Dst: \" << ROI() ";
            }
        }

        cpp << ");" << std::endl;

        cpp << "}" << std::endl << std::endl;

        if (!ok)
        {
            aFailedFunctions.push_back(elem);
        }
    }

    header << headerFooter;
    cpp << cppFooter;
}