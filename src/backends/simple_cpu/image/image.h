#pragma once

#include "imageView.h"
#include <common/defines.h>
#include <common/fileIO/filetypes/tiffFile.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/image/sizePitched.h>
#include <common/safeCast.h>
#include <cstddef>
#include <filesystem>
#include <vector>

namespace mpp::image::cpuSimple
{

template <PixelType T> class Image : public ImageView<T>
{
  private:
    std::vector<T> mData;

  public:
    Image() = delete;
    Image(int aWidth, int aHeight) : Image(Size2D(aWidth, aHeight))
    {
    }
    Image(const Size2D &aSize) : ImageView<T>(aSize), mData(aSize.TotalSize())
    {
        ImageView<T>::PointerRef()    = mData.data();
        ImageView<T>::PointerRoiRef() = mData.data();
        ImageView<T>::PitchRef()      = sizeof(T) * to_size_t(aSize.x);
    }
    ~Image() override
    {
        ImageView<T>::PointerRef()    = nullptr;
        ImageView<T>::PointerRoiRef() = nullptr;
        ImageView<T>::PitchRef()      = 0;
        ImageView<T>::ROIRef()        = Roi();
        ImageView<T>::SizeAllocRef()  = Size2D();
    }

    Image(const Image &) = delete;
    Image(Image &&aOther) noexcept
    {
        ImageView<T>::PointerRef()    = aOther.PointerRef();
        ImageView<T>::PointerRoiRef() = aOther.PointerRoiRef();
        ImageView<T>::PitchRef()      = aOther.PitchRef();
        ImageView<T>::ROIRef()        = aOther.ROIRef();
        ImageView<T>::SizeAllocRef()  = aOther.SizeAllocRef();
        mData                         = std::move(aOther.mData);

        aOther.PointerRef()    = nullptr;
        aOther.PointerRoiRef() = nullptr;
        aOther.PitchRef()      = 0;
        aOther.ROIRef()        = Roi();
        aOther.SizeAllocRef()  = Size2D();
    }

    Image &operator=(const Image &) = delete;
    Image &operator=(Image &&aOther) noexcept
    {
        if (std::addressof(aOther) == std::addressof(*this))
        {
            return *this;
        }
        ImageView<T>::PointerRef()    = aOther.PointerRef();
        ImageView<T>::PointerRoiRef() = aOther.PointerRoiRef();
        ImageView<T>::PitchRef()      = aOther.PitchRef();
        ImageView<T>::ROIRef()        = aOther.ROIRef();
        ImageView<T>::SizeAllocRef()  = aOther.SizeAllocRef();
        mData                         = std::move(aOther.mData);

        aOther.PointerRef()    = nullptr;
        aOther.PointerRoiRef() = nullptr;
        aOther.PitchRef()      = 0;
        aOther.ROIRef()        = Roi();
        aOther.SizeAllocRef()  = Size2D();
        return *this;
    }

    static bool CanLoad(const std::filesystem::path &aFilename)
    {
        mpp::fileIO::TIFFFile tiff(aFilename);
        const bool ok = tiff.TryToOpenAndReadHeader();
        if (!ok)
        {
            return false;
        }

        return tiff.GetDataType() == mpp::image::pixel_type_enum<T>::pixelType;
    }

    static Image Load(const std::filesystem::path &aFilename)
    {
        mpp::fileIO::TIFFFile tiff(aFilename);
        tiff.OpenAndRead();

        if (tiff.GetDataType() != mpp::image::pixel_type_enum<T>::pixelType)
        {
            throw FILEIOEXCEPTION(aFilename, "Failed to read TIFF-image. Expected pixel type: "
                                                 << mpp::image::pixel_type_enum<T>::pixelType
                                                 << " but TIFF file is of type " << tiff.GetDataType());
        }

        Image<T> img(tiff.SizePlane());
        std::copy(tiff.Pixels<T>(), tiff.Pixels<T>() + tiff.SizePlane().TotalSize(), img.mData.data());

        return img;
    }

    static std::vector<Image> LoadPlanar(const std::filesystem::path &aFilename)
    {
        mpp::fileIO::TIFFFile tiff(aFilename);
        tiff.OpenAndRead();

        if (tiff.GetDataType() != mpp::image::pixel_type_enum<T>::pixelType)
        {
            throw FILEIOEXCEPTION(aFilename, "Failed to read TIFF-image. Expected pixel type: "
                                                 << mpp::image::pixel_type_enum<T>::pixelType
                                                 << " but TIFF file is of type " << tiff.GetDataType());
        }

        if (!tiff.IsPlanar())
        {
            throw FILEIOEXCEPTION(aFilename, "The provided TIFF-file is not a multi-color planar image file.");
        }

        const size_t colorChannels = to_size_t(tiff.SamplesPerPixel());

        std::vector<Image> ret;

        for (size_t i = 0; i < colorChannels; i++)
        {
            ret.emplace_back(tiff.SizePlane());
            std::copy(tiff.Pixels<T>() + tiff.SizePlane().TotalSize() * i,
                      tiff.Pixels<T>() + tiff.SizePlane().TotalSize() * i + tiff.SizePlane().TotalSize(),
                      ret[i].mData.data());
        }

        return ret;
    }

    void Save(const std::filesystem::path &aFilename)
    {
        mpp::fileIO::TIFFFile::WriteTIFF(aFilename, ImageView<T>::SizeAllocRef().x, ImageView<T>::SizeAllocRef().y, 0.0,
                                         mpp::image::pixel_type_enum<T>::pixelType, mData.data(), 9);
    }
};
} // namespace mpp::image::cpuSimple