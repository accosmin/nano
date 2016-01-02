#include "unit_test.hpp"
#include "thread/pool.h"
#include "thread/thread.h"
#include "math/random.hpp"
#include "math/epsilon.hpp"
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

NANOCV_CASE(evaluate)
{
        typedef double scalar_t;

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

NANOCV_END_MODULE()

