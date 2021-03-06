#include "utest.h"
#include <numeric>
#include "core/tpool.h"
#include "core/random.h"

using namespace nano;

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
                        assert(begin < end);
                        assert(0u <= begin);
                        assert(end <= size);
                        NANO_UNUSED1_RELEASE(size);
                        for (size_t i = begin; i < end; ++ i)
                        {
                                results[i] = op(i);
                        }
                });

                return std::accumulate(results.begin(), results.end(), tscalar(0));
        }
}

NANO_BEGIN_MODULE(test_core_tpool)

NANO_CASE(empty)
{
        auto& pool = tpool_t::instance();

        NANO_CHECK_EQUAL(pool.workers(), nano::physical_cpus());
        NANO_CHECK_EQUAL(pool.tasks(), 0u);
}

NANO_CASE(enqueue)
{
        auto& pool = tpool_t::instance();

        NANO_CHECK_EQUAL(pool.workers(), nano::physical_cpus());
        NANO_CHECK_EQUAL(pool.tasks(), 0u);

        const size_t max_tasks = 1024;
        const auto tasks = urand<size_t>(1u, max_tasks, make_rng());

        std::vector<size_t> tasks_done;

        std::mutex mutex;
        {
                tpool_section_t<future_t> futures;
                for (size_t j = 0; j < tasks; ++ j)
                {
                        futures.push_back(pool.enqueue([=, &mutex, &tasks_done]()
                        {
                                const auto sleep1 = urand<size_t>(1, 5, make_rng());
                                std::this_thread::sleep_for(std::chrono::milliseconds(sleep1));

                                {
                                        const std::lock_guard<std::mutex> lock(mutex);

                                        tasks_done.push_back(j + 1);
                                }
                        }));
                }
        }

        NANO_CHECK_EQUAL(pool.workers(), nano::physical_cpus());
        NANO_CHECK_EQUAL(pool.tasks(), 0u);

        NANO_CHECK_EQUAL(tasks_done.size(), tasks);
        for (size_t j = 0; j < tasks; ++ j)
        {
                NANO_CHECK(std::find(tasks_done.begin(), tasks_done.end(), j + 1) != tasks_done.end());
        }
}

NANO_CASE(evaluate)
{
        const size_t min_size = 1;
        const size_t max_size = 9 * 9 * 9 * 1;

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

                for (size_t chunk = 1; chunk < 8; ++ chunk)
                {
                        const auto mt = test_mt<scalar_t>(size, chunk, op);
                        NANO_CHECK_CLOSE(st, mt, nano::epsilon1<scalar_t>());
                }
        }
}

NANO_END_MODULE()
