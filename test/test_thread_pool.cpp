#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_thread_pool"

#include <boost/test/unit_test.hpp>
#include "thread/pool.h"
#include "thread/thread.h"
#include "math/random.hpp"

BOOST_AUTO_TEST_CASE(test_thread_pool_empty)
{
        thread::pool_t pool;

        BOOST_CHECK_EQUAL(pool.n_workers(), thread::n_threads());
        BOOST_CHECK_EQUAL(pool.n_tasks(), 0);
}

BOOST_AUTO_TEST_CASE(test_thread_pool_enqueue)
{
        thread::pool_t pool;

        const size_t n_tests = 7;
        const size_t n_max_jobs = pool.n_workers() * 16;

        for (size_t t = 0; t < n_tests; ++ t)
        {
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

                BOOST_CHECK_EQUAL(pool.n_workers(), thread::n_threads());
                BOOST_CHECK_EQUAL(pool.n_tasks(), 0);

                BOOST_CHECK_EQUAL(tasks_done.size(), n_tasks);
                for (size_t j = 0; j < n_tasks; ++ j)
                {
                        BOOST_CHECK(std::find(tasks_done.begin(), tasks_done.end(), j + 1) != tasks_done.end());
                }
        }
}
