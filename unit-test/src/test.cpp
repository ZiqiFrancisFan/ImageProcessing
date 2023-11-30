#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "SignalProcessing.h"
#include <iostream>

TEST_CASE("trial")
{
    std::cout << "Hello!" << std::endl;
}

TEST_CASE("Signal Processing")
{
    Signal1D<float> signal(128, {0, 127});
}


