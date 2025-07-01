#pragma once
#include "testBase.h"
#include <common/image/border.h>
#include <vector>

void runPixel32sC1(size_t aIterations, size_t aRepeats, int aWidth, int aHeight, const mpp::image::Border &aBorder,
                   std::vector<mpp::TestResult> &aTestResult, mpp::TestsToRun aTestsToRun);