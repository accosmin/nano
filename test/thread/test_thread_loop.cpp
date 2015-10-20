#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_thread_loop"

#include <boost/test/unit_test.hpp>
#include "math/abs.hpp"
#include "thread/thread.h"
#include "thread/loopi.hpp"
#include "math/epsilon.hpp"
#include <cmath>
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
                for (size_t i = 0; i < results.size(); i ++)
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
                thread::loopi(size, [results = std::ref(results), op = op] (size_t i)
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
                thread::loopi(size, nthreads, [results = std::ref(results), op = op] (size_t i)
                {
                        results.get()[i] = op(i);
                });

                return std::accumulate(results.begin(), results.end(), tscalar(0));
        }
}

BOOST_AUTO_TEST_CASE(test_thread_loop)
{
        const size_t min_size = 37;
        const size_t max_size = 1023;

        typedef double scalar_t;

        // operator to test
        const auto op = [](size_t i)
        {
                const auto ii = static_cast<scalar_t>(i);
                const auto vi = std::cos(ii) + std::sin(ii);

                auto temp = scalar_t(0);
                for (int j = 0; j < 16; j ++)
                {
                        temp += j * std::tan(-ii + j);
                }

                return vi * temp / 32;
        };

        // test for different problems size
        for (size_t size = min_size; size <= max_size; size *= 3)
        {
                // single-threaded
                const scalar_t st = test_st<scalar_t>(size, op);

                // multi-threaded
                const scalar_t mt = test_mt<scalar_t>(size, op);
                BOOST_CHECK_LE(math::abs(st - mt), math::epsilon1<scalar_t>());

                for (size_t nthreads = thread::n_threads(); nthreads <= thread::max_n_threads(); nthreads ++)
                {
                        const scalar_t mtx = test_mt<scalar_t>(size, nthreads, op);
                        BOOST_CHECK_LE(math::abs(st - mtx), math::epsilon1<scalar_t>());
                }
        }
}
