#include "bohachevsky.h"
#include <cmath>

using namespace nano;

function_bohachevsky1_t::function_bohachevsky1_t() :
        function_t("Bohachevsky1", 2, 2, 2, convexity::no, 100)
{
}

scalar_t function_bohachevsky1_t::vgrad(const vector_t& x, vector_t* gx) const
{
        const auto x1 = x(0);
        const auto x2 = x(1);

        const auto pi = std::atan2(scalar_t(0.0), scalar_t(-0.0));
        const auto p1 = 3 * pi * x1;
        const auto p2 = 4 * pi * x2;

        const auto u = x1 * x1 + 2 * x2 * x2;

        if (gx)
        {
                (*gx)(0) = 2 * x1 + scalar_t(0.9) * std::sin(p1) * pi;
                (*gx)(1) = 4 * x2 + scalar_t(1.6) * std::sin(p2) * pi;
        }

        return u - scalar_t(0.3) * std::cos(p1) - scalar_t(0.4) * std::cos(p2) + scalar_t(0.7);
}

function_bohachevsky2_t::function_bohachevsky2_t() :
        function_t("Bohachevsky2", 2, 2, 2, convexity::no, 100)
{
}

scalar_t function_bohachevsky2_t::vgrad(const vector_t& x, vector_t* gx) const
{
        const auto x1 = x(0);
        const auto x2 = x(1);

        const auto pi = std::atan2(scalar_t(0.0), scalar_t(-0.0));
        const auto p1 = 3 * pi * x1;
        const auto p2 = 4 * pi * x2;

        const auto u = x1 * x1 + 2 * x2 * x2;

        if (gx)
        {
                (*gx)(0) = 2 * x1 + scalar_t(0.9) * std::sin(p1) * pi * std::cos(p2);
                (*gx)(1) = 4 * x2 + scalar_t(1.2) * std::sin(p2) * pi * std::cos(p1);
        }

        return u - scalar_t(0.3) * std::cos(p1) * std::cos(p2) + scalar_t(0.3);
}

function_bohachevsky3_t::function_bohachevsky3_t() :
        function_t("Bohachevsky3", 2, 2, 2, convexity::no, 100)
{
}

scalar_t function_bohachevsky3_t::vgrad(const vector_t& x, vector_t* gx) const
{
        const auto x1 = x(0);
        const auto x2 = x(1);

        const auto pi = std::atan2(scalar_t(0.0), scalar_t(-0.0));
        const auto p1 = 3 * pi * x1;
        const auto p2 = 4 * pi * x2;

        const auto u = x1 * x1 + 2 * x2 * x2;

        if (gx)
        {
                (*gx)(0) = 2 * x1 + scalar_t(0.9) * std::sin(p1 + p2) * pi;
                (*gx)(1) = 4 * x2 + scalar_t(1.2) * std::sin(p1 + p2) * pi;
        }

        return u - scalar_t(0.3) * std::cos(p1 + p2) + scalar_t(0.3);
}
