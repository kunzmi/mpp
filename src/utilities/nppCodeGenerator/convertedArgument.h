#pragma once
#include <string>
#include <utilities/nppParser/function.h>
#include <vector>

namespace opp::utilities::nppParser
{
class ConvertedFunction;

class ConvertedArgument
{
  public:
    ConvertedArgument(const ConvertedFunction &aFunction, std::string aType, std::string aName);
    ConvertedArgument(const Argument *aArgument, const ConvertedFunction &aFunction);
    ConvertedArgument(const Argument *aArgument, ConvertedArgument *aLinkedArgument,
                      const ConvertedFunction &aFunction);
    ~ConvertedArgument() = default;

    ConvertedArgument(const ConvertedArgument &)     = default;
    ConvertedArgument(ConvertedArgument &&) noexcept = default;

    ConvertedArgument &operator=(const ConvertedArgument &)     = default;
    ConvertedArgument &operator=(ConvertedArgument &&) noexcept = default;

    std::string ToStringDeclaration() const;
    std::string ToStringNppCall() const;

    const Argument *SrcArgument() const
    {
        return mArgument;
    }

    const std::string &Type() const
    {
        return mType;
    }
    std::string &Type()
    {
        return mType;
    }

    const ConvertedArgument *LinkedArgument() const
    {
        return mLinkedArgument;
    }
    ConvertedArgument *&LinkedArgument()
    {
        return mLinkedArgument;
    }

    const std::string &Name() const
    {
        return mName;
    }
    std::string &Name()
    {
        return mName;
    }

    const std::string &Call() const
    {
        return mCall;
    }
    std::string &Call()
    {
        return mCall;
    }

    const bool &IsSkippedInDeclaration() const
    {
        return mIsSkippedInDeclaration;
    }
    bool &IsSkippedInDeclaration()
    {
        return mIsSkippedInDeclaration;
    }

    const bool &IsSkippedInCall() const
    {
        return mIsSkippedInCall;
    }
    bool &IsSkippedInCall()
    {
        return mIsSkippedInCall;
    }

    bool IsInputImage() const;

    bool IsOutputImage() const;

    bool IsMask() const;

    bool IsStep() const;

  private:
    const Argument *mArgument{nullptr};
    ConvertedArgument *mLinkedArgument{nullptr};
    const ConvertedFunction &mFunction;
    std::string mType;
    std::string mName;
    std::string mCall;
    bool mIsSkippedInDeclaration{false};
    bool mIsSkippedInCall{true};
    bool mIsInputImage{false};
    bool mIsOutputImage{false};

    static std::string ConvertNppType(const std::string &aNPPType);
    static std::string GetImageType(const std::string &aNPPType, int aChannelCount);
};

} // namespace opp::utilities::nppParser