#pragma once
#include "bigTiffImageFileDirectory.h"
#include "tiffFile.h"

namespace opp::fileIO::bigTiffTag
{
template <class T>
BigIFDEntry<T>::BigIFDEntry(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigImageFileDirectoryEntry(aFile, aTagID)
{
    if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 8)
    {
        ulong64 temp = mTag.Offset.UInt64Val; // NOLINT

        mValue.resize(mTag.Count);
        auto *vals = reinterpret_cast<T *>(&temp);

        if (IsLittleEndian())
        {
            for (size_t i = 0; i < mTag.Count; i++)
            {
                mValue[i] = vals[i]; // NOLINT(clang-analyzer-core.uninitialized.Assign)
            }
        }
        else
        {
            for (size_t i = 0; i < mTag.Count; i++)
            {
                mValue[i] = vals[8 / GetTiffTypeSizeInBytes(mTag.Type) - 1 -
                                 i]; // NOLINT(clang-analyzer-core.uninitialized.Assign)
            }
        }
    }
    else
    {
        const size_t currentOffset = TellRead();
        SeekRead(mTag.Offset.UInt64Val, std::ios_base::beg); // NOLINT
        mValue = ReadLE<T>(mTag.Count);
        SeekRead(currentOffset, std::ios_base::beg);
    }
}

BigIFDEntry<std::string>::BigIFDEntry(TIFFFile &aFile, ushort aTagID) // NOLINT(misc-definitions-in-headers)
    : File("", aFile.IsLittleEndian()), BigImageFileDirectoryEntry(aFile, aTagID)
{
    if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 8)
    {
        ulong64 temp = mTag.Offset.UInt64Val; // NOLINT

        std::vector<char> text(mTag.Count + 1, 0);
        char *vals = reinterpret_cast<char *>(&temp); // NOLINT -> vararg??

        if (IsLittleEndian())
        {
            for (size_t i = 0; i < mTag.Count; i++)
            {
                text[i] = vals[i]; // NOLINT(clang-analyzer-core.uninitialized.Assign)
            }
        }
        else
        {
            for (size_t i = 0; i < mTag.Count; i++)
            {
                text[i] = vals[8 - 1 - i]; // NOLINT(clang-analyzer-core.uninitialized.Assign)
            }
        }
        text[mTag.Count] = 0;
        mValue           = std::string(text.data());
    }
    else
    {
        const size_t currentOffset = TellRead();
        SeekRead(mTag.Offset.UInt64Val, std::ios_base::beg); // NOLINT
        mValue = ReadString(mTag.Count);
        SeekRead(currentOffset, std::ios_base::beg);
    }
}

const std::string &BigIFDEntry<std::string>::Value() const // NOLINT(misc-definitions-in-headers)
{
    return mValue;
}
} // namespace opp::fileIO::bigTiffTag