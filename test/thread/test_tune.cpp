#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_tune"

#include <boost/test/unit_test.hpp>
#include "math/abs.hpp"
#include "thread/pool.h"
#include "thread/thread.h"
#include "math/random.hpp"
#include "math/epsilon.hpp"
#include "min/tune_log10_mt.hpp"

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
                std::tie(stfx, stx) = min::tune_log10(op, minlog, maxlog, epslog, splits);

                // multi-threaded version
                thread::pool_t pool(splits);
                tscalar mtfx, mtx;
                std::tie(mtfx, mtx) = min::tune_log10_mt(op, pool, minlog, maxlog, epslog, splits);

                const tscalar epsilon = math::epsilon2<tscalar>();

                // check optimum result
                BOOST_CHECK_LE(math::abs(stfx - b), epsilon);
                BOOST_CHECK_LE(math::abs(mtfx - b), epsilon);

                // check optimum parameters
                BOOST_CHECK_LE(math::abs(stx - a), epsilon);
                BOOST_CHECK_LE(math::abs(mtx - a), epsilon);
        }
}

BOOST_AUTO_TEST_CASE(test_tune)
{
        typedef double scalar_t;

        const size_t n_tests = 16;
        const scalar_t minlog = -6.0;
        const scalar_t maxlog = +6.0;
        const scalar_t epslog = math::epsilon2<scalar_t>();
        const size_t splits = thread::n_threads();

        for (size_t t = 0; t < n_tests; t ++)
        {
                math::random_t<scalar_t> agen(+0.1, +1.0);
                math::random_t<scalar_t> bgen(-2.0, +2.0);

                test::check(agen(), bgen(), minlog, maxlog, epslog, splits);
        }
}

