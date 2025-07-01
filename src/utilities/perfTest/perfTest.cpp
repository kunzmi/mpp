#include "common/exception.h"
#include <algorithm>
#include <cmath>
#include <common/image/border.h>
#include <common/version.h>
#include <cstddef>
#include <fstream>
#include <ios>
#include <iostream>
#include <numeric>
#include <vector>

#include "pixel8uC1.h"
#include "pixel8uC3.h"
#include "pixel8uC4.h"

#include "pixel16uC1.h"
#include "pixel16uC3.h"
#include "pixel16uC4.h"

#include "pixel32fC1.h"
#include "pixel32fC3.h"
#include "pixel32fC4.h"

#include "pixel32sC1.h"
#include "testBase.h"

using namespace mpp;
using namespace mpp::image;

int main(int argc, char *argv[])
{
    try
    {
        std::cout << "Performance comparison for " << MPP_PROJECT_NAME << " version " << MPP_VERSION << "."
                  << std::endl;

        size_t iterations = 100;
        size_t repeats    = 10;
        int imgWidth      = 4096;
        int imgHeight     = 4096;
        Border b          = Border(0, 0, 0, 0);
        TestsToRun testsToRun;
        std::vector<mpp::TestResult> results;

        std::cout << "argC: " << argc << std::endl;
        for (size_t i = 1; i < static_cast<size_t>(argc); i++)
        {
            int value = 0;
            std::stringstream ss(argv[i]);
            ss >> value;
            std::string vals(argv[i]);

            if (i == 1)
            {
                iterations = static_cast<size_t>(value);
                std::cout << "iterations: " << iterations << std::endl;
            }
            if (i == 2)
            {
                repeats = static_cast<size_t>(value);
                std::cout << "repeats: " << repeats << std::endl;
            }
            if (i == 3)
            {
                imgWidth = value;
                std::cout << "imgWidth: " << imgWidth << std::endl;
            }
            if (i == 4)
            {
                imgHeight = value;
                std::cout << "imgHeight: " << imgHeight << std::endl;
            }
            if (i == 5)
            {
                b.lowerX = value;
            }
            if (i == 6)
            {
                b.higherX = value;
            }
            if (i == 7)
            {
                b.lowerY = value;
            }
            if (i == 8)
            {
                b.higherY = value;
            }
            if (i >= 9)
            {
                std::cout << "vals: '" << vals << "'" << std::endl;
                if (vals == "noA")
                {
                    testsToRun.Arithmetic = false;
                }
                if (vals == "noD")
                {
                    testsToRun.DataExchange = false;
                }
                if (vals == "noF")
                {
                    testsToRun.Filtering = false;
                }
                if (vals == "noG")
                {
                    testsToRun.GeometryTransform = false;
                }
                if (vals == "noM")
                {
                    testsToRun.Morphology = false;
                }
                if (vals == "noS")
                {
                    testsToRun.Statistics = false;
                }
                if (vals == "noT")
                {
                    testsToRun.Threshold = false;
                }
            }
        }

        std::cout << "Image size: " << imgWidth << " x " << imgHeight << std::endl;
        std::cout << "ROI border: " << b << std::endl;

        runPixel8uC1(iterations, repeats, imgWidth, imgHeight, b, results, testsToRun);
        std::cout << std::endl << std::endl << "----------------------" << std::endl << std::endl << std::endl;
        runPixel8uC3(iterations, repeats, imgWidth, imgHeight, b, results, testsToRun);
        std::cout << std::endl << std::endl << "----------------------" << std::endl << std::endl << std::endl;
        runPixel8uC4(iterations, repeats, imgWidth, imgHeight, b, results, testsToRun);
        std::cout << std::endl << std::endl << "----------------------" << std::endl << std::endl << std::endl;

        runPixel16uC1(iterations, repeats, imgWidth, imgHeight, b, results, testsToRun);
        std::cout << std::endl << std::endl << "----------------------" << std::endl << std::endl << std::endl;
        runPixel16uC3(iterations, repeats, imgWidth, imgHeight, b, results, testsToRun);
        std::cout << std::endl << std::endl << "----------------------" << std::endl << std::endl << std::endl;
        runPixel16uC4(iterations, repeats, imgWidth, imgHeight, b, results, testsToRun);
        std::cout << std::endl << std::endl << "----------------------" << std::endl << std::endl << std::endl;

        runPixel32fC1(iterations, repeats, imgWidth, imgHeight, b, results, testsToRun);
        std::cout << std::endl << std::endl << "----------------------" << std::endl << std::endl << std::endl;
        runPixel32fC3(iterations, repeats, imgWidth, imgHeight, b, results, testsToRun);
        std::cout << std::endl << std::endl << "----------------------" << std::endl << std::endl << std::endl;
        runPixel32fC4(iterations, repeats, imgWidth, imgHeight, b, results, testsToRun);
        std::cout << std::endl << std::endl << "----------------------" << std::endl << std::endl << std::endl;
        // runPixel8uC1(iterations, repeats, imgWidth, imgHeight, -1);

        // runPixel32sC1(iterations, repeats, imgWidth, imgHeight, b, results, testsToRun);
        // std::cout << std::endl << std::endl << "----------------------" << std::endl << std::endl << std::endl;

        std::sort(results.begin(), results.end());

        std::ofstream csv("results.csv", std::ofstream::out);
        csv << "Image size: " << imgWidth << " x " << imgHeight << " Border: " << b << std::endl;
        csv << "Name" << "\t";
        csv << "TotalMPP" << "\t";
        csv << "TotalNPP" << "\t";
        csv << "MeanMPP" << "\t";
        csv << "MeanNPP" << "\t";
        csv << "StdMPP" << "\t";
        csv << "StdNPP" << "\t";
        csv << "MinMPP" << "\t";
        csv << "MinNPP" << "\t";
        csv << "MaxMPP" << "\t";
        csv << "MaxNPP" << "\t";
        csv << "AbsoluteDifferenceMSec" << "\t";
        csv << "RelativeDifference" << std::endl;
        for (const auto &elem : results)
        {
            csv << elem;
        }

        // Image<Pixel16uC1> ttt(imgWidth, imgHeight);
        // nv::Image16uC3 ttt2(imgWidth, imgHeight);
        //  ttt2.Max()
        //      ttt.M()
    }
    catch (mpp::MPPException &ex)
    {
        std::cout << ex.what();
    }
    catch (...)
    {
        return 1;
    }
    return 0;
}
