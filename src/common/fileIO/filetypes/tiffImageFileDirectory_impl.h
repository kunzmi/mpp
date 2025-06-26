#pragma once
#include "tiffFile.h"
#include "tiffImageFileDirectory.h"
#include <common/defines.h>
#include <common/fileIO/file.h>
#include <common/safeCast.h>
#include <cstring>
#include <ios>
#include <ostream>
#include <string>
#include <vector>

namespace mpp::fileIO::tiffTag
{
template <class T>
IFDEntry<T>::IFDEntry(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), ImageFileDirectoryEntry(aFile, aTagID)
{
    if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 4)
    {
        uint temp = mTag.Offset.UIntVal; // NOLINT
        mValue.resize(mTag.Count);
        auto *vals = reinterpret_cast<T *>(&temp);

        if (IsLittleEndian())
        {
            for (size_t i = 0; i < mTag.Count && i < 4; i++)
            {
                mValue[i] = vals[i]; // NOLINT(clang-analyzer-core.uninitialized.Assign)
            }
        }
        else
        {
            for (size_t i = 0; i < mTag.Count && i < 4; i++)
            {
                mValue[i] =                                          // NOLINT(clang-analyzer-core.uninitialized.Assign)
                    vals[4 / GetTiffTypeSizeInBytes(mTag.Type) - 1 - // NOLINT(clang-analyzer-core.uninitialized.Assign)
                         i];                                         // NOLINT(clang-analyzer-core.uninitialized.Assign)
            }
        }
    }
    else
    {
        const size_t currentOffset = TellRead();
        SeekRead(mTag.Offset.UIntVal, std::ios_base::beg); // NOLINT
        mValue = ReadLE<T>(mTag.Count);
        SeekRead(currentOffset, std::ios_base::beg);
    }
}
template <class T>
IFDEntry<T>::IFDEntry(T aValue, ushort aTagID, TiffType aFieldType)
    : File("", true), ImageFileDirectoryEntry(aTagID, aFieldType, 1)
{
    mValue.push_back(aValue);
}

template <typename T>
IFDEntry<T>::IFDEntry(std::vector<T> &&aValues, ushort aTagID, TiffType aFieldType)
    : File("", true), ImageFileDirectoryEntry(aTagID, aFieldType, to_uint(aValues.size())), mValue(std::move(aValues))
{
}

template <class T> void IFDEntry<T>::SavePass1(std::ostream &aStream)
{
    if (sizeof(T) * mValue.size() <= 4)
    {
        char temp[4] = {0};
        std::memcpy(reinterpret_cast<char *>(temp), mValue.data(), sizeof(T) * mValue.size());
        const uint val = *(reinterpret_cast<uint *>(temp));

        WriteEntryHeader(val, aStream, to_int(mValue.size()));
        mOffsetInStream = 0;
    }
    else
    {
        mOffsetInStream = WriteEntryHeader(0, aStream, to_int(mValue.size()));
    }
}

template <class T> void IFDEntry<T>::SavePass2(std::ostream &aStream)
{
    if (mOffsetInStream != 0)
    {
        WritePass2(reinterpret_cast<char *>(mValue.data()), mValue.size() * sizeof(T), aStream);
    }
}

IFDEntry<std::string>::IFDEntry(TIFFFile &aFile, ushort aTagID) // NOLINT(misc-definitions-in-headers)
    : File("", aFile.IsLittleEndian()), ImageFileDirectoryEntry(aFile, aTagID)
{
    if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 4)
    {
        uint temp = mTag.Offset.UIntVal; // NOLINT

        std::vector<char> text(to_size_t(mTag.Count) + 1, 0);
        char *vals = reinterpret_cast<char *>(&temp); // NOLINT -> vararg??

        if (IsLittleEndian())
        {
            for (size_t i = 0; i < mTag.Count && i < 4; i++)
            {
                text[i] = vals[i];
            }
        }
        else
        {
            for (size_t i = 0; i < mTag.Count && i < 4; i++)
            {
                text[i] = vals[4 - 1 - i];
            }
        }
        text[mTag.Count] = 0;
        mValue           = std::string(text.data());
    }
    else
    {
        const size_t currentOffset = TellRead();
        SeekRead(mTag.Offset.UIntVal, std::ios_base::beg); // NOLINT
        mValue = ReadString(mTag.Count);
        SeekRead(currentOffset, std::ios_base::beg);
    }
}

IFDEntry<std::string>::IFDEntry(const std::string &aValue, ushort aTagID) // NOLINT(misc-definitions-in-headers)
    : File("", true), ImageFileDirectoryEntry(aTagID, TiffType::ASCII, to_uint(aValue.size() + 1)), mValue(aValue)
{
}

const std::string &IFDEntry<std::string>::Value() const // NOLINT(misc-definitions-in-headers)
{
    return mValue;
}

void IFDEntry<std::string>::SavePass1(std::ostream &aStream) // NOLINT(misc-definitions-in-headers)
{
    if (mValue.size() < 3) // last element in array should be 0
    {
        mValue.resize(4, 0); // make sure that we have at least 4 characters
        const uint val = *(reinterpret_cast<const uint *>(mValue.c_str()));
        WriteEntryHeader(val, aStream, to_int(mValue.size()));
        mOffsetInStream = 0;
    }
    else
    {
        mOffsetInStream = WriteEntryHeader(0, aStream, to_int(mValue.size()) + 1); // with trailing '0'
    }
}

void IFDEntry<std::string>::SavePass2(std::ostream &aStream) // NOLINT(misc-definitions-in-headers)
{
    if (mOffsetInStream != 0)
    {
        WritePass2(mValue.c_str(), mValue.size() + 1, aStream); // with trailing '0'
    }
}
} // namespace mpp::fileIO::tiffTag