#include "utest.h"
#include "thread/pool.h"
#include "math/random.h"

using namespace nano;

NANO_BEGIN_MODULE(test_thread_pool)

NANO_CASE(empty)
{
        auto& pool = thread_pool_t::instance();

        NANO_CHECK_EQUAL(pool.workers(), nano::logical_cpus());
        NANO_CHECK_EQUAL(pool.active_workers(), nano::logical_cpus());
        NANO_CHECK_EQUAL(pool.tasks(), 0u);
}

NANO_CASE(enqueue)
{
        auto& pool = thread_pool_t::instance();

        const size_t threads = nano::logical_cpus();
        const size_t max_tasks = threads * 16;

        for (size_t active_workers = 1; active_workers <= threads; ++ active_workers)
        {
                pool.activate(active_workers);

                NANO_CHECK_EQUAL(pool.workers(), threads);
                NANO_CHECK_EQUAL(pool.active_workers(), active_workers);
                NANO_CHECK_EQUAL(pool.tasks(), 0u);

                nano::random_t<size_t> rnd(1, max_tasks);
                const size_t tasks = rnd();

                std::vector<size_t> tasks_done;

                std::mutex mutex;
                {
                        section_t<future_t> futures;
                        for (size_t j = 0; j < tasks; ++ j)
                        {
                                futures.push_back(pool.enqueue([=, &mutex, &tasks_done]()
                                {
                                        const size_t sleep1 = nano::random_t<size_t>(1, 5)();
                                        std::this_thread::sleep_for(std::chrono::milliseconds(sleep1));

                                        {
                                                const std::lock_guard<std::mutex> lock(mutex);

                                                tasks_done.push_back(j + 1);
                                        }
                                }));
                        }
                }

                NANO_CHECK_EQUAL(pool.workers(), threads);
                NANO_CHECK_EQUAL(pool.active_workers(), active_workers);
                NANO_CHECK_EQUAL(pool.tasks(), 0u);

                NANO_CHECK_EQUAL(tasks_done.size(), tasks);
                for (size_t j = 0; j < tasks; ++ j)
                {
                        NANO_CHECK(std::find(tasks_done.begin(), tasks_done.end(), j + 1) != tasks_done.end());
                }
        }
}

NANO_END_MODULE()
