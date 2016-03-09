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
        tscalar test_st(const size_t size, toperator op)
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
        tscalar test_mt(const size_t size, toperator op)
        {
                std::vector<tscalar> results(size);
                zob::loopi(size, [results = std::ref(results), op = op] (size_t i)
                {
                        results.get()[i] = op(i);
                });

                return std::accumulate(results.begin(), results.end(), tscalar(0));
        }

        // multi-threaded (variable number of threads)
        template
        <
                typename tscalar,
                typename toperator
        >
        tscalar test_mt(const size_t size, const size_t nthreads, toperator op)
        {
                std::vector<tscalar> results(size);
                zob::loopi(size, nthreads, [results = std::ref(results), op = op] (size_t i)
                {
                        results.get()[i] = op(i);
                });

                return std::accumulate(results.begin(), results.end(), tscalar(0));
        }
}

ZOB_BEGIN_MODULE(test_thread_loop)

ZOB_CASE(evaluate)
{
        const size_t min_size = 7;
        const size_t max_size = 3 * 3 * 7;

        typedef double scalar_t;

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
                const scalar_t mt = test_mt<scalar_t>(size, op);
                ZOB_CHECK_CLOSE(st, mt, zob::epsilon1<scalar_t>());

                for (size_t nthreads = 1; nthreads <= zob::n_threads(); nthreads += 2)
                {
                        const scalar_t mtx = test_mt<scalar_t>(size, nthreads, op);
                        ZOB_CHECK_CLOSE(st, mtx, zob::epsilon1<scalar_t>());
                }
        }
}

ZOB_END_MODULE()
