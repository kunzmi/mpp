// Creates the class members for Npp::ImageView

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <json.h>
#include <map>
#include <sstream>
#include <string>
#include <utilities/nppParser/function.h>
#include <utilities/nppParser/nppParser.h>
#include <vector>

using namespace opp::utilities::nppParser;

void process8uC1(std::vector<Function> &aFunctions, std::vector<Function> &aFailedFunctions);

int main()
{
    std::vector<Function> undecided;

    std::vector<Function> functions = NPPParser::GetFunctions();

    std::vector<Function> functionsNoCtxDoublets;

    for (auto &elem : functions)
    {
        if (NPPParser::IsCtx(elem))
        {
            functionsNoCtxDoublets.push_back(elem);
        }
        else
        {
            const std::string ctxName = elem.name + "_Ctx";

            auto iter = std::find_if(functions.begin(), functions.end(),
                                     [&ctxName](const Function &aFunc) { return aFunc.name == ctxName; });
            if (iter == functions.end())
            {
                // function has no _Ctx counterpart so we have to keep it
                functionsNoCtxDoublets.push_back(elem);
            }
        }
    }

    std::vector<Function> failed;

    process8uC1(functionsNoCtxDoublets, failed);

    std::cout << "Failed on " << failed.size() << " functions.";
    try
    {
    }
    catch (...)
    {
        return 1;
    }
    return 0;
}
