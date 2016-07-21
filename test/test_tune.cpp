#include "scalar.h"
#include "utest.hpp"
#include "math/tune.hpp"
#include "math/random.hpp"
#include "math/epsilon.hpp"

using namespace nano;

static void check(const scalar_t a, const scalar_t b,
        const scalar_t minlog, const scalar_t maxlog, const scalar_t eps)
{
        const auto log10_space = make_log10_space(minlog, maxlog, eps);

        const auto min = std::pow(scalar_t(10), minlog);
        const auto max = std::pow(scalar_t(10), maxlog);
        const auto linear_space = make_linear_space(min, max, eps);

        const auto epsilon = epsilon3<scalar_t>();

        const auto op1 = [a=a, b=b] (const scalar_t x)
        {
                return (x - a) * (x - a) + b;
        };
        {
                const auto ret = tune(op1, log10_space);

                NANO_CHECK_CLOSE(ret.optimum(), b, epsilon);
                NANO_CHECK_CLOSE(ret.param0(), a, epsilon);
        }
        {
                const auto ret = tune(op1, linear_space);

                NANO_CHECK_CLOSE(ret.optimum(), b, epsilon);
                NANO_CHECK_CLOSE(ret.param0(), a, epsilon);
        }

        const auto op2 = [a=a, b=b] (const scalar_t x, const scalar_t y)
        {
                return (x - b) * (x - b) + (y - a) * (y - a) + b;
        };
        {
                const auto ret = tune(op2, log10_space, log10_space);

                NANO_CHECK_CLOSE(ret.optimum(), b, epsilon);
                NANO_CHECK_CLOSE(ret.param0(), b, epsilon);
                NANO_CHECK_CLOSE(ret.param1(), a, epsilon);
        }
        {
                const auto ret = tune(op2, linear_space, log10_space);

                NANO_CHECK_CLOSE(ret.optimum(), b, epsilon);
                NANO_CHECK_CLOSE(ret.param0(), b, epsilon);
                NANO_CHECK_CLOSE(ret.param1(), a, epsilon);
        }
        {
                const auto ret = tune(op2, log10_space, linear_space);

                NANO_CHECK_CLOSE(ret.optimum(), b, epsilon);
                NANO_CHECK_CLOSE(ret.param0(), b, epsilon);
                NANO_CHECK_CLOSE(ret.param1(), a, epsilon);
        }
        {
                const auto ret = tune(op2, linear_space, linear_space);

                NANO_CHECK_CLOSE(ret.optimum(), b, epsilon);
                NANO_CHECK_CLOSE(ret.param0(), b, epsilon);
                NANO_CHECK_CLOSE(ret.param1(), a, epsilon);
        }
}

NANO_BEGIN_MODULE(test_tune)

NANO_CASE(tune_grid)
{
        const auto n_tests = 16;
        const auto minlog = scalar_t(-6.0);
        const auto maxlog = scalar_t(+6.0);
        const auto epslog = epsilon0<scalar_t>();

        for (auto t = 0; t < n_tests; ++ t)
        {
                random_t<scalar_t> agen(scalar_t(+0.1), scalar_t(+1.0));
                random_t<scalar_t> bgen(scalar_t(+0.2), scalar_t(+2.0));

                check(agen(), bgen(), minlog, maxlog, epslog);
        }
}

NANO_CASE(tune_finite)
{
        const auto op1 = [] (const auto param1)
        {
                return param1 * param1;
        };
        const auto op2 = [&] (const auto param1, const auto param2)
        {
                return op1(param1) + op1(param2);
        };
        const auto op3 = [&] (const auto param1, const auto param2, const auto param3)
        {
                return op2(param1, param2) + op1(param3);
        };
        const auto op4 = [&] (const auto param1, const auto param2, const auto param3, const auto param4)
        {
                return op3(param1, param2, param3) + op1(param4);
        };

        const auto params1 = make_finite_space(0, 1);
        const auto params2 = make_finite_space(3, 2, 1);
        const auto params3 = make_finite_space(2, 3, 4, 5);
        const auto params4 = make_finite_space(7, 6, 5, 4, 3);

        const auto ret1 = tune(op1, params1);
        const auto ret2 = tune(op2, params1, params2);
        const auto ret3 = tune(op3, params1, params2, params3);
        const auto ret4 = tune(op4, params1, params2, params3, params4);

        NANO_CHECK_EQUAL(ret1.optimum(), op1(0));
        NANO_CHECK_EQUAL(ret1.param0(), 0);

        NANO_CHECK_EQUAL(ret2.optimum(), op2(0, 1));
        NANO_CHECK_EQUAL(ret2.param0(), 0);
        NANO_CHECK_EQUAL(ret2.param1(), 1);

        NANO_CHECK_EQUAL(ret3.optimum(), op3(0, 1, 2));
        NANO_CHECK_EQUAL(ret3.param0(), 0);
        NANO_CHECK_EQUAL(ret3.param1(), 1);
        NANO_CHECK_EQUAL(ret3.param2(), 2);

        NANO_CHECK_EQUAL(ret4.optimum(), op4(0, 1, 2, 3));
        NANO_CHECK_EQUAL(ret4.param0(), 0);
        NANO_CHECK_EQUAL(ret4.param1(), 1);
        NANO_CHECK_EQUAL(ret4.param2(), 2);
        NANO_CHECK_EQUAL(ret4.param3(), 3);
}

NANO_END_MODULE()

