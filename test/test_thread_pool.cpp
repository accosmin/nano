#include "unit_test.hpp"
#include "thread/pool.h"
#include "thread/thread.h"
#include "math/random.hpp"

NANO_BEGIN_MODULE(test_thread_pool)

NANO_CASE(empty)
{
        nano::pool_t pool;

        const size_t n_threads = nano::n_threads();
        const size_t n_active_workers = n_threads;

        NANO_CHECK_EQUAL(pool.n_workers(), n_threads);
        NANO_CHECK_EQUAL(pool.n_active_workers(), n_active_workers);
        NANO_CHECK_EQUAL(pool.n_jobs(), 0);
}

NANO_CASE(enqueue)
{
        nano::pool_t pool;

        const size_t n_threads = nano::n_threads();
        const size_t n_max_jobs = n_threads * 16;

        for (size_t n_active_workers = 1; n_active_workers <= n_threads; ++ n_active_workers)
        {
                pool.activate(n_active_workers);

                NANO_CHECK_EQUAL(pool.n_workers(), n_threads);
                NANO_CHECK_EQUAL(pool.n_active_workers(), n_active_workers);
                NANO_CHECK_EQUAL(pool.n_jobs(), 0);

                nano::random_t<size_t> rnd(1, n_max_jobs);
                const size_t n_tasks = rnd();

                std::vector<size_t> tasks_done;

                std::mutex mutex;

                for (size_t j = 0; j < n_tasks; ++ j)
                {
                        pool.enqueue([=, &mutex, &tasks_done]()
                        {
                                const size_t sleep1 = nano::random_t<size_t>(1, 5)();
                                std::this_thread::sleep_for(std::chrono::milliseconds(sleep1));

                                {
                                        const std::lock_guard<std::mutex> lock(mutex);

                                        tasks_done.push_back(j + 1);
                                }
                        });
                }

                pool.wait();

                NANO_CHECK_EQUAL(pool.n_workers(), n_threads);
                NANO_CHECK_EQUAL(pool.n_active_workers(), n_active_workers);
                NANO_CHECK_EQUAL(pool.n_jobs(), 0);

                NANO_CHECK_EQUAL(tasks_done.size(), n_tasks);
                for (size_t j = 0; j < n_tasks; ++ j)
                {
                        NANO_CHECK(std::find(tasks_done.begin(), tasks_done.end(), j + 1) != tasks_done.end());
                }
        }
}

NANO_END_MODULE()
