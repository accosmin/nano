#include "unit_test.hpp"
#include "thread/thread.h"
#include "thread/loopi.hpp"
#include "math/epsilon.hpp"
#include <numeric>

namespace
{
        // single-threaded
        template
        <
                typename tscalar,
                typename toperator
        >
        tscalar test_st(const size_t size, const toperator op)
        {
                std::vector<tscalar> results(size);
                for (size_t i = 0; i < results.size(); ++ i)
                {
                        results[i] = op(i);
                }

                return std::accumulate(results.begin(), results.end(), tscalar(0));
        }

        // multi-threaded (default number of threads)
        template
        <
                typename tscalar,
                typename toperator
        >
        tscalar test_mt(const size_t size, const toperator op, const size_t splits)
        {
                std::vector<tscalar> results(size);
                thread::loopi(size, [results = std::ref(results), op = op] (size_t i)
                {
                        results.get()[i] = op(i);
                }, splits);

                return std::accumulate(results.begin(), results.end(), tscalar(0));
        }

        // multi-threaded (variable number of threads)
        template
        <
                typename tscalar,
                typename toperator
        >
        tscalar test_mt(const size_t size, const size_t nthreads, const toperator op, const size_t splits)
        {
                std::vector<tscalar> results(size);
                thread::loopi(size, nthreads, [results = std::ref(results), op = op] (size_t i)
                {
                        results.get()[i] = op(i);
                }, splits);

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
        const auto op = [](size_t i)
        {
                const auto ii = static_cast<scalar_t>(i);
                return ii * ii + 1 - ii;
        };

        // test for different problems size
        for (size_t size = min_size; size <= max_size; size *= 3)
        {
                // single-threaded
                const scalar_t st = test_st<scalar_t>(size, op);

                // multi-threaded
                const scalar_t mt = test_mt<scalar_t>(size, op, 1);
                NANO_CHECK_CLOSE(st, mt, nano::epsilon1<scalar_t>());

                for (size_t nthreads = 1; nthreads <= thread::concurrency(); nthreads += 2)
                {
                        for (size_t splits = 1; splits <= 8; ++ splits)
                        {
                                const scalar_t mtx = test_mt<scalar_t>(size, nthreads, op, splits);
                                NANO_CHECK_CLOSE(st, mtx, nano::epsilon1<scalar_t>());
                        }
                }
        }
}

NANO_END_MODULE()
