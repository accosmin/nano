#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_log_search"

#include <boost/test/unit_test.hpp>
#include "libnanocv/types.h"
#include "libnanocv/util/abs.hpp"
#include "libnanocv/util/random.hpp"
#include "libnanocv/util/epsilon.hpp"
#include "libnanocv/util/thread_pool.h"
#include "libnanocv/util/log_search.hpp"

namespace test
{
        using namespace ncv;

        void check(scalar_t a, scalar_t b, scalar_t minlog, scalar_t maxlog, scalar_t epslog, size_t splits)
        {
                auto op = [=] (scalar_t x)
                {
                        return (x - a) * (x - a) + b;
                };

                // single-threaded version
                const std::pair<scalar_t, scalar_t> ret1 = ncv::log10_min_search(op, minlog, maxlog, epslog, splits);

                // multi-threaded version
                thread_pool_t pool(splits);
                const std::pair<scalar_t, scalar_t> retx = ncv::log10_min_search_mt(op, pool, minlog, maxlog, epslog, splits);

                const scalar_t epsilon = math::epsilon2<scalar_t>();

                // check optimum result
                BOOST_CHECK_LE(math::abs(ret1.first - b), epsilon);
                BOOST_CHECK_LE(math::abs(retx.first - b), epsilon);

                // check optimum parameters
                BOOST_CHECK_LE(math::abs(ret1.second - a), epsilon);
                BOOST_CHECK_LE(math::abs(retx.second - a), epsilon);
        }
}

BOOST_AUTO_TEST_CASE(test_log_search)
{
        using namespace ncv;

        const size_t n_tests = 16;
        const scalar_t minlog = -6.0;
        const scalar_t maxlog = +6.0;
        const scalar_t epslog = math::epsilon2<scalar_t>();
        const size_t splits = ncv::n_threads();

        for (size_t t = 0; t < n_tests; t ++)
        {
                random_t<scalar_t> agen(+0.1, +1.0);
                random_t<scalar_t> bgen(-2.0, +2.0);

                test::check(agen(), bgen(), minlog, maxlog, epslog, splits);
        }
}

