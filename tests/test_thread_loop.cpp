#include "utest.h"
#include "thread/loopi.h"
#include "math/epsilon.h"
#include <numeric>

namespace
{
        // single-threaded
        template <typename tscalar, typename toperator>
        tscalar test_st(const size_t size, const toperator op)
        {
                std::vector<tscalar> results(size);
                for (size_t i = 0; i < results.size(); ++ i)
                {
                        results[i] = op(i);
                }

                return std::accumulate(results.begin(), results.end(), tscalar(0));
        }

        // multi-threaded
        template <typename tscalar, typename toperator>
        tscalar test_mt(const size_t size, const size_t chunk, const toperator op)
        {
                std::vector<tscalar> results(size);
                nano::loopi(size, chunk, [&results = results, size = size, op = op] (const size_t begin, const size_t end)
                {
                        NANO_CHECK_LESS(begin, end);
                        NANO_CHECK_LESS_EQUAL(0u, begin);
                        NANO_CHECK_LESS_EQUAL(end, size);
                        for (size_t i = begin; i < end; ++ i)
                        {
                                results[i] = op(i);
                        }
                });

                return std::accumulate(results.begin(), results.end(), tscalar(0));
        }
}

NANO_BEGIN_MODULE(test_thread_loop)

NANO_CASE(evaluate)
{
        const size_t min_size = 7;
        const size_t max_size = 3 * 3 * 7;

        using scalar_t = double;

        // operator to test
        const auto op = [](const size_t i)
        {
                const auto ii = static_cast<scalar_t>(i);
                return ii * ii + 1 - ii;
        };

        // test for different problems size
        for (size_t size = min_size; size <= max_size; size *= 3)
        {
                // single-threaded
                const auto st = test_st<scalar_t>(size, op);

                // multi-threaded
                for (size_t nthreads = 1; nthreads <= nano::logical_cpus(); nthreads += 2)
                {
                        nano::thread_pool_t::instance().activate(nthreads);

                        for (size_t chunk = 1; chunk < 8; ++ chunk)
                        {
                                const auto mt = test_mt<scalar_t>(size, chunk, op);
                                NANO_CHECK_CLOSE(st, mt, nano::epsilon1<scalar_t>());
                        }
                }
        }
}

NANO_END_MODULE()
