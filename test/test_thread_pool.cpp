#include "unit_test.hpp"
#include "thread/pool.h"
#include "thread/thread.h"
#include "math/random.hpp"

NANOCV_BEGIN_MODULE(test_thread_pool)

NANOCV_CASE(empty)
{
        thread::pool_t pool;

        const size_t n_threads = thread::n_threads();
        const size_t n_active_workers = n_threads;

        NANOCV_CHECK_EQUAL(pool.n_workers(), n_threads);
        NANOCV_CHECK_EQUAL(pool.n_active_workers(), n_active_workers);
        NANOCV_CHECK_EQUAL(pool.n_tasks(), 0);
}

NANOCV_CASE(enqueue)
{
        thread::pool_t pool;

        const size_t n_threads = thread::n_threads();
        const size_t n_max_jobs = n_threads * 16;

        for (size_t n_active_workers = 1; n_active_workers <= n_threads; ++ n_active_workers)
        {
                pool.activate(n_active_workers);

                NANOCV_CHECK_EQUAL(pool.n_workers(), n_threads);
                NANOCV_CHECK_EQUAL(pool.n_active_workers(), n_active_workers);
                NANOCV_CHECK_EQUAL(pool.n_tasks(), 0);

                math::random_t<size_t> rnd(1, n_max_jobs);
                const size_t n_tasks = rnd();

                std::vector<size_t> tasks_done;

                std::mutex mutex;

                for (size_t j = 0; j < n_tasks; ++ j)
                {
                        pool.enqueue([=, &mutex, &tasks_done]()
                        {
                                const size_t sleep1 = math::random_t<size_t>(1, 5)();
                                std::this_thread::sleep_for(std::chrono::milliseconds(sleep1));

                                {
                                        const std::lock_guard<std::mutex> lock(mutex);

                                        tasks_done.push_back(j + 1);
                                }
                        });
                }

                pool.wait();

                NANOCV_CHECK_EQUAL(pool.n_workers(), n_threads);
                NANOCV_CHECK_EQUAL(pool.n_active_workers(), n_active_workers);
                NANOCV_CHECK_EQUAL(pool.n_tasks(), 0);

                NANOCV_CHECK_EQUAL(tasks_done.size(), n_tasks);
                for (size_t j = 0; j < n_tasks; ++ j)
                {
                        NANOCV_CHECK(std::find(tasks_done.begin(), tasks_done.end(), j + 1) != tasks_done.end());
                }
        }
}

NANOCV_END_MODULE()
