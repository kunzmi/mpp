// nppParser.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//

#include "function.h"
#include "nppParser.h"
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <json.h>
#include <map>
#include <sstream>
#include <string>
#include <vector>

bool baseNameHas_(const std::string &aFName)
{
    const size_t first_  = aFName.find('_');
    const size_t second_ = aFName.find('_', first_ + 1);

    std::string shouldBeType = aFName.substr(first_ + 1, second_ - first_ - 1);
    // if the type string doesn't start with a number, it is likely that the actual name contains a '_'

    if (shouldBeType.empty())
    {
        return false;
    }
    if (shouldBeType[0] == '8' || shouldBeType[0] == '1' || shouldBeType[0] == '3' || shouldBeType[0] == '6')
    {
        return false;
    }
    return true;
}

bool isMergedColorTwist(const std::string &aFName)
{
    return aFName.find("_ColorTwist32f_") < aFName.size();
}

std::string getTypeString(const std::string &aFName)
{
    if (aFName == "nppiYCCKToCMYK_JPEG_601_8u_P4R" || aFName == "nppiYCCKToCMYK_JPEG_601_8u_P4R_Ctx")
    {
        return "8u";
    }

    if (!baseNameHas_(aFName))
    {
        const size_t first_  = aFName.find('_');
        const size_t second_ = aFName.find('_', first_ + 1);
        return aFName.substr(first_ + 1, second_ - first_ - 1);
    }
    const size_t first_  = aFName.find('_');
    const size_t second_ = aFName.find('_', first_ + 1);
    const size_t third_  = aFName.find('_', second_ + 1);
    return aFName.substr(second_ + 1, third_ - second_ - 1);
}

std::string getChannelString(const std::string &aFName)
{
    if (aFName == "nppiYCCKToCMYK_JPEG_601_8u_P4R" || aFName == "nppiYCCKToCMYK_JPEG_601_8u_P4R_Ctx")
    {
        return "P4R";
    }

    if (!baseNameHas_(aFName) && !isMergedColorTwist(aFName))
    {
        const size_t first_  = aFName.find('_');
        const size_t second_ = aFName.find('_', first_ + 1);
        const size_t third_  = aFName.find('_', second_ + 1);
        return aFName.substr(second_ + 1, third_ - second_ - 1);
    }

    const size_t first_  = aFName.find('_');
    const size_t second_ = aFName.find('_', first_ + 1);
    const size_t third_  = aFName.find('_', second_ + 1);
    const size_t fourth_ = aFName.find('_', third_ + 1);
    return aFName.substr(third_ + 1, fourth_ - third_ - 1);
}

std::string getBaseName(const std::string &aFName)
{
    if (aFName == "nppiYCCKToCMYK_JPEG_601_8u_P4R" || aFName == "nppiYCCKToCMYK_JPEG_601_8u_P4R_Ctx")
    {
        return "nppiYCCKToCMYK_JPEG_601";
    }
    if (aFName == "nppiFusedAbsDiff_Threshold_GTVal_Ctx" || aFName == "nppiFusedAbsDiff_Threshold_GTVal_I_Ctx")
    {
        return "nppiFusedAbsDiff_Threshold_GTVal";
    }

    if (baseNameHas_(aFName))
    {
        const size_t first_  = aFName.find('_');
        const size_t second_ = aFName.find('_', first_ + 1);
        return aFName.substr(0, second_);
    }

    return aFName.substr(0, aFName.find('_'));
}

bool getContext(const std::string &aFName)
{
    return aFName.substr(aFName.size() - 3) == "Ctx";
}

struct NppiFunction
{
    std::string name;
    std::string category;
    bool type8s{false};
    bool type8u{false};
    bool type16s{false};
    bool type16sc{false};
    bool type16u{false};
    bool type16f{false};
    bool type32s{false};
    bool type32sc{false};
    bool type32u{false};
    bool type32f{false};
    bool type32fc{false};
    bool type64f{false};

    bool channel1{false};
    bool channel2{false};
    bool channel3{false};
    bool channel4{false};
    bool channel4A{false};

    bool regular{false};
    bool inplace{false};

    bool nonCtx{false};
    bool withCtx{false};

    bool nonSfs{false};
    bool withSfs{false};

    bool nonMask{false};
    bool withMask{false};

    bool planarInSomeWay{false};
};

inline void to_json(nlohmann::json &aj, const NppiFunction &aFunction)
{
    aj = nlohmann::json{{"name", aFunction.name},           {"category", aFunction.category},
                        {"type8s", aFunction.type8s},       {"type8u", aFunction.type8u},
                        {"type16s", aFunction.type16s},     {"type16sc", aFunction.type16sc},
                        {"type16u", aFunction.type16u},     {"type16f", aFunction.type16f},
                        {"type32s", aFunction.type32s},     {"type32sc", aFunction.type32sc},
                        {"type32u", aFunction.type32u},     {"type32f", aFunction.type32f},
                        {"type32fc", aFunction.type32fc},   {"type64f", aFunction.type64f},
                        {"channel1", aFunction.channel1},   {"channel2", aFunction.channel2},
                        {"channel3", aFunction.channel3},   {"channel4", aFunction.channel4},
                        {"channel4A", aFunction.channel4A}, {"regular", aFunction.regular},
                        {"inplace", aFunction.inplace},     {"nonCtx", aFunction.nonCtx},
                        {"withCtx", aFunction.withCtx},     {"nonSfs", aFunction.nonSfs},
                        {"withSfs", aFunction.withSfs},     {"nonMask", aFunction.nonMask},
                        {"withMask", aFunction.withMask},   {"planarInSomeWay", aFunction.planarInSomeWay}};
}

inline std::string to_string(const NppiFunction &aFunction, const std::string &aSeperator = "\t")
{
    std::stringstream ss;

    ss << aFunction.name << aSeperator;
    ss << aFunction.category << aSeperator;
    ss << (aFunction.type8s ? "true" : "") << aSeperator;
    ss << (aFunction.type8u ? "true" : "") << aSeperator;
    ss << (aFunction.type16s ? "true" : "") << aSeperator;
    ss << (aFunction.type16sc ? "true" : "") << aSeperator;
    ss << (aFunction.type16u ? "true" : "") << aSeperator;
    ss << (aFunction.type16f ? "true" : "") << aSeperator;
    ss << (aFunction.type32s ? "true" : "") << aSeperator;
    ss << (aFunction.type32sc ? "true" : "") << aSeperator;
    ss << (aFunction.type32u ? "true" : "") << aSeperator;
    ss << (aFunction.type32f ? "true" : "") << aSeperator;
    ss << (aFunction.type32fc ? "true" : "") << aSeperator;
    ss << (aFunction.type64f ? "true" : "") << aSeperator;
    ss << (aFunction.channel1 ? "true" : "") << aSeperator;
    ss << (aFunction.channel2 ? "true" : "") << aSeperator;
    ss << (aFunction.channel3 ? "true" : "") << aSeperator;
    ss << (aFunction.channel4 ? "true" : "") << aSeperator;
    ss << (aFunction.channel4A ? "true" : "") << aSeperator;
    ss << (aFunction.regular ? "true" : "") << aSeperator;
    ss << (aFunction.inplace ? "true" : "") << aSeperator;
    ss << (aFunction.nonCtx ? "true" : "") << aSeperator;
    ss << (aFunction.withCtx ? "true" : "") << aSeperator;
    ss << (aFunction.nonSfs ? "true" : "") << aSeperator;
    ss << (aFunction.withSfs ? "true" : "") << aSeperator;
    ss << (aFunction.nonMask ? "true" : "") << aSeperator;
    ss << (aFunction.withMask ? "true" : "") << aSeperator;
    ss << (aFunction.planarInSomeWay ? "true" : "");

    return ss.str();
}

inline std::string getTitleString(const std::string &aSeperator = "\t")
{
    std::stringstream ss;

    ss << "name" << aSeperator;
    ss << "category" << aSeperator;
    ss << "type8s" << aSeperator;
    ss << "type8u" << aSeperator;
    ss << "type16s" << aSeperator;
    ss << "type16sc" << aSeperator;
    ss << "type16u" << aSeperator;
    ss << "type16f" << aSeperator;
    ss << "type32s" << aSeperator;
    ss << "type32sc" << aSeperator;
    ss << "type32u" << aSeperator;
    ss << "type32f" << aSeperator;
    ss << "type32fc" << aSeperator;
    ss << "type64f" << aSeperator;
    ss << "channel1" << aSeperator;
    ss << "channel2" << aSeperator;
    ss << "channel3" << aSeperator;
    ss << "channel4" << aSeperator;
    ss << "channel4A" << aSeperator;
    ss << "regular" << aSeperator;
    ss << "inplace" << aSeperator;
    ss << "nonCtx" << aSeperator;
    ss << "withCtx" << aSeperator;
    ss << "nonSfs" << aSeperator;
    ss << "withSfs" << aSeperator;
    ss << "nonMask" << aSeperator;
    ss << "withMask" << aSeperator;
    ss << "planarInSomeWay";

    return ss.str();
}

bool setType(const std::string &aTypeString, NppiFunction &aFunction)
{
    if (aTypeString.substr(0, 2) == "8s")
    {
        aFunction.type8s = true;
        return true;
    }
    if (aTypeString.substr(0, 2) == "8u")
    {
        aFunction.type8u = true;
        return true;
    }
    if (aTypeString.substr(0, 3) == "16f")
    {
        aFunction.type16f = true;
        return true;
    }
    if (aTypeString.substr(0, 4) == "16sc")
    {
        aFunction.type16sc = true;
        return true;
    }
    if (aTypeString.substr(0, 3) == "16s")
    {
        aFunction.type16s = true;
        return true;
    }
    if (aTypeString.substr(0, 3) == "16u")
    {
        aFunction.type16u = true;
        return true;
    }
    if (aTypeString.substr(0, 4) == "32sc")
    {
        aFunction.type32sc = true;
        return true;
    }
    if (aTypeString.substr(0, 3) == "32s")
    {
        aFunction.type32s = true;
        return true;
    }
    if (aTypeString.substr(0, 3) == "32u")
    {
        aFunction.type32u = true;
        return true;
    }
    if (aTypeString.substr(0, 4) == "32fc")
    {
        aFunction.type32fc = true;
        return true;
    }
    if (aTypeString.substr(0, 3) == "32f")
    {
        aFunction.type32f = true;
        return true;
    }
    if (aTypeString.substr(0, 3) == "64f")
    {
        aFunction.type64f = true;
        return true;
    }
    return false;
}

bool isInplace(const std::string &aChannelString)
{
    return aChannelString.find('I') < aChannelString.size();
}

bool isMasked(const std::string &aChannelString)
{
    return aChannelString.find('M') < aChannelString.size();
}

bool isSfs(const std::string &aChannelString)
{
    return aChannelString.find("Sfs") < aChannelString.size();
}

bool setChannel(const std::string &aChannelString, NppiFunction &aFunction)
{
    bool ok               = false;
    const bool is_masked  = isMasked(aChannelString);
    const bool is_inplace = isInplace(aChannelString);
    const bool is_sfs     = isSfs(aChannelString);

    std::string str = aChannelString;
    str.erase(std::remove(str.begin(), str.end(), 'I'), str.end());
    str.erase(std::remove(str.begin(), str.end(), 'M'), str.end());
    str.erase(std::remove(str.begin(), str.end(), 'S'), str.end());
    str.erase(std::remove(str.begin(), str.end(), 'f'), str.end());
    str.erase(std::remove(str.begin(), str.end(), 's'), str.end());

    if (str == "C1R" || str == "AC1R" || str == "C1" || str == "C1CR" || str == "C1C2R" || str == "C1C3R" ||
        str == "C1C4R" || str == "C1AC4R")
    {
        aFunction.channel1 = true;
        ok                 = true;
    }
    else if (str == "C2R" || str == "C2" || str == "C2CR" || str == "C2C1R" || str == "C2C3R" || str == "C2C4R")
    {
        aFunction.channel2 = true;
        ok                 = true;
    }
    else if (str == "C3R" || str == "C3" || str == "C3CR" || str == "C3C1R" || str == "C3C2R" || str == "C3C4R" ||
             str == "C3A0C4R")
    {
        aFunction.channel3 = true;
        ok                 = true;
    }
    else if (str == "C4R" || str == "C4" || str == "C4CR" || str == "C4C1R" || str == "C4C3R")
    {
        aFunction.channel4 = true;
        ok                 = true;
    }
    else if (str == "AC4R" || str == "AC4CR" || str == "AC4C1R" || str == "AC4C2R")
    {
        aFunction.channel4A = true;
        ok                  = true;
    }
    else if (str == "P1R")
    {
        aFunction.channel1        = true;
        aFunction.planarInSomeWay = true;
        ok                        = true;
    }
    else if (str == "P2R" || str == "P2P3R" || str == "C2P2R" || str == "C2P3R" || str == "P2C2R" || str == "P2C3R" ||
             str == "P2C4R")
    {
        aFunction.channel2        = true;
        aFunction.planarInSomeWay = true;
        ok                        = true;
    }
    else if (str == "P3R" || str == "P3P2R" || str == "C3P3R" || str == "C3P2R" || str == "P3C2R" || str == "P3C3R" ||
             str == "P3C4R" || str == "P3AC4R")
    {
        aFunction.channel3        = true;
        aFunction.planarInSomeWay = true;
        ok                        = true;
    }
    else if (str == "P4R" || str == "P4P3R" || str == "C4P4R" || str == "C4P3R" || str == "P4C3R" || str == "P4C4R")
    {
        aFunction.channel4        = true;
        aFunction.planarInSomeWay = true;
        ok                        = true;
    }
    else if (str == "AP4R" || str == "AP4C4R" || str == "AC4P3R" || str == "AC4P4R")
    {
        aFunction.channel4A       = true;
        aFunction.planarInSomeWay = true;
        ok                        = true;
    }

    if (ok)
    {
        if (is_masked)
        {
            aFunction.withMask = true;
        }
        else
        {
            aFunction.nonMask = true;
        }

        if (is_inplace)
        {
            aFunction.inplace = true;
        }
        else
        {
            aFunction.regular = true;
        }

        if (is_sfs)
        {
            aFunction.withSfs = true;
        }
        else
        {
            aFunction.nonSfs = true;
        }
    }
    return ok;
}

bool setContext(bool aCtx, NppiFunction &aFunction)
{
    if (!aCtx)
    {
        aFunction.nonCtx = true;
        return true;
    }
    if (aCtx)
    {
        aFunction.withCtx = true;
        return true;
    }
    return false;
}

int main()
{
    try
    {
        std::map<std::string, NppiFunction> nppiFunctions;
        // std::vector<nppParser::Function> undecided;

        const std::vector<nppParser::Function> functions = nppParser::NPPParser::GetFunctions();

        // std::set<std::string> undecidedClean;

        for (const auto &elem : functions)
        {
            const std::string name = getBaseName(elem.name);

            NppiFunction &f = nppiFunctions[name];
            f.name          = name;
            f.category      = elem.category;
            bool ok         = setType(getTypeString(elem.name), f);
            ok &= setChannel(getChannelString(elem.name), f);
            ok &= setContext(getContext(elem.name), f);
            if (!ok)
            {
                /*undecided.push_back(elem);
                undecidedClean.insert(name);
                continue;*/
            }

            /*NppiFunction& f = nppiFunctions[name];
            f.name = name;
            setType(getTypeString(elem.name), f);
            setChannel(getChannelString(elem.name), f);
            setContext(getContext(elem.name), f);*/
        }

        // std::sort(undecided.begin(), undecided.end(), [](const nppParser::Function& a, const nppParser::Function& b)
        // {return a.name < b.name; });

        // for (const auto& elem : undecidedClean)
        //{
        //	//if (undecidedClean.find(elem.name) == undecidedClean.end())
        //	{
        //		std::cout << elem << std::endl;
        //	}
        // }
        // std::cout << std::endl;
        // std::cout << std::endl;

        std::ofstream of(R"(F:\ogpp\nppParser\functionlist.txt)");

        of << getTitleString() << std::endl;

        for (const auto &[name, function] : nppiFunctions)
        {
            of << to_string(function) << std::endl;
        }
        of.close();
    }
    catch (...)
    {
        return 1;
    }
    return 0;
}
