#pragma once

#include <json.h>
#include <string>
#include <vector>

namespace mpp::utilities::nppParser
{
struct Argument
{
    std::string type;
    std::string name;

    bool operator==(const Argument &aOther) const
    {
        return type == aOther.type && name == aOther.name;
    }
};

struct Function
{
    std::string returnType;
    std::string name;
    std::vector<Argument> arguments;
    std::string category;

    bool operator==(const Function &aOther) const
    {
        if (arguments.size() != aOther.arguments.size())
        {
            return false;
        }
        for (size_t i = 0; i < arguments.size(); i++)
        {
            if (arguments[i] != aOther.arguments[i])
            {
                return false;
            }
        }
        return returnType == aOther.returnType && name == aOther.name && category == aOther.category;
    }
};

inline void to_json(nlohmann::json &aj, const Argument &aArgument)
{
    aj = nlohmann::json{{"type", aArgument.type}, {"name", aArgument.name}};
}

inline void from_json(const nlohmann::json &aj, Argument &aArgument)
{
    aj.at("type").get_to(aArgument.type);
    aj.at("name").get_to(aArgument.name);
}

inline void to_json(nlohmann::json &aj, const Function &aFunction)
{
    aj = nlohmann::json{{"returnType", aFunction.returnType},
                        {"name", aFunction.name},
                        {"arguments", aFunction.arguments},
                        {"category", aFunction.category}};
}

inline void from_json(const nlohmann::json &aj, Function &aFunction)
{
    aj.at("returnType").get_to(aFunction.returnType);
    aj.at("name").get_to(aFunction.name);
    aj.at("arguments").get_to(aFunction.arguments);
    aj.at("category").get_to(aFunction.category);
}
} // namespace mpp::utilities::nppParser