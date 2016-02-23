#include "unit_test.hpp"
#include "thread/pool.h"
#include "thread/thread.h"
#include "math/random.hpp"
#include "math/epsilon.hpp"
#include "math/tune_fixed.hpp"
#include "math/tune_log10_mt.hpp"

namespace test
{
        template
        <
                typename tscalar
        >
        void check(tscalar a, tscalar b, tscalar minlog, tscalar maxlog, tscalar epslog, size_t splits)
        {
                auto op = [=] (tscalar x)
                {
                        return (x - a) * (x - a) + b;
                };

                // single-threaded version
                tscalar stfx, stx;
                std::tie(stfx, stx) = math::tune_log10(op, minlog, maxlog, epslog, splits);

                // multi-threaded version
                thread::pool_t pool(splits);
                tscalar mtfx, mtx;
                std::tie(mtfx, mtx) = math::tune_log10_mt(op, pool, minlog, maxlog, epslog, splits);

                const tscalar epsilon = math::epsilon2<tscalar>();

                // check optimum result
                NANOCV_CHECK_CLOSE(stfx, b, epsilon);
                NANOCV_CHECK_CLOSE(mtfx, b, epsilon);

                // check optimum parameters
                NANOCV_CHECK_CLOSE(stx, a, epsilon);
                NANOCV_CHECK_CLOSE(mtx, a, epsilon);
        }
}

NANOCV_BEGIN_MODULE(test_tune)

NANOCV_CASE(tune_log10)
{
        using scalar_t = double;

        const size_t n_tests = 16;
        const scalar_t minlog = -6.0;
        const scalar_t maxlog = +6.0;
        const scalar_t epslog = math::epsilon2<scalar_t>();
        const size_t splits = thread::n_threads();

        for (size_t t = 0; t < n_tests; ++ t)
        {
                math::random_t<scalar_t> agen(+0.1, +1.0);
                math::random_t<scalar_t> bgen(-2.0, +2.0);

                test::check(agen(), bgen(), minlog, maxlog, epslog, splits);
        }
}

NANOCV_CASE(tune_fixed)
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

        const auto params1 = { 0, 1 };
        const auto params2 = { 3, 2, 1 };
        const auto params3 = { 2, 3, 4, 5 };
        const auto params4 = { 7, 6, 5, 4, 3 };

        const auto ret1 = math::tune_fixed(op1, params1);
        const auto ret2 = math::tune_fixed(op2, params1, params2);
        const auto ret3 = math::tune_fixed(op3, params1, params2, params3);
        const auto ret4 = math::tune_fixed(op4, params1, params2, params3, params4);

        NANOCV_CHECK_EQUAL(std::get<0>(ret1), op1(0));
        NANOCV_CHECK_EQUAL(std::get<1>(ret1), 0);

        NANOCV_CHECK_EQUAL(std::get<0>(ret2), op2(0, 1));
        NANOCV_CHECK_EQUAL(std::get<1>(ret2), 0);
        NANOCV_CHECK_EQUAL(std::get<2>(ret2), 1);

        NANOCV_CHECK_EQUAL(std::get<0>(ret3), op3(0, 1, 2));
        NANOCV_CHECK_EQUAL(std::get<1>(ret3), 0);
        NANOCV_CHECK_EQUAL(std::get<2>(ret3), 1);
        NANOCV_CHECK_EQUAL(std::get<3>(ret3), 2);

        NANOCV_CHECK_EQUAL(std::get<0>(ret4), op4(0, 1, 2, 3));
        NANOCV_CHECK_EQUAL(std::get<1>(ret4), 0);
        NANOCV_CHECK_EQUAL(std::get<2>(ret4), 1);
        NANOCV_CHECK_EQUAL(std::get<3>(ret4), 2);
        NANOCV_CHECK_EQUAL(std::get<4>(ret4), 3);
}

NANOCV_END_MODULE()

