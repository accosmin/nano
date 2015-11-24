#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_thread_pool"

#include <boost/test/unit_test.hpp>
#include "thread/pool.h"
#include "thread/thread.h"
#include "math/random.hpp"
#include <iostream>

BOOST_AUTO_TEST_CASE(test_thread_pool)
{
        thread::pool_t pool;

        std::mutex mutex;

        // check that there is no job to do
        BOOST_CHECK_EQUAL(pool.n_workers(), thread::n_threads());
        BOOST_CHECK_EQUAL(pool.n_tasks(), 0);

        const size_t n_tests = 8;
        const size_t n_max_jobs = pool.n_workers() * 16;

        // run multiple tests ...
        for (size_t t = 0; t < n_tests; ++ t)
        {
                math::random_t<size_t> rnd(1, n_max_jobs);

                // ... enqueue jobs
                const size_t n_tasks = rnd();
                std::cout << "@pool [" << (t + 1) << "/" << n_tests
                          << "]: creating " << n_tasks << " jobs ...\n";

                std::vector<size_t> tasks_done;

                for (size_t j = 0; j < n_tasks; ++ j)
                {
                        pool.enqueue([=, &mutex, &tasks_done]()
                        {
                                const size_t sleep1 = math::random_t<size_t>(5, 10)();
                                std::this_thread::sleep_for(std::chrono::milliseconds(sleep1));

                                {
                                        const std::lock_guard<std::mutex> lock(mutex);

                                        std::cout << "#job [" << (j + 1) << "/" << n_tasks << "@"
                                                  << (t + 1) << "/" << n_tests << "] started ...\n";
                                }

                                const size_t sleep2 = math::random_t<size_t>(5, 20)();
                                std::this_thread::sleep_for(std::chrono::milliseconds(sleep2));

                                {
                                        const std::lock_guard<std::mutex> lock(mutex);

                                        std::cout << "#job [" << (j + 1) << "/" << n_tasks << "@"
                                                   << (t + 1) << "/" << n_tests << "] done.\n";

                                        tasks_done.push_back(j + 1);
                                }
                        });
                }

                // ... wait for all jobs to finish
                std::cout << "@pool [" << (t + 1) << "/" << n_tests
                          << "]: waiting for " << pool.n_tasks() << " jobs ...\n";

                pool.wait();

                std::cout << "@pool [" << (t + 1) << "/" << n_tests
                          << "]: waiting done (enqueued " << pool.n_tasks() << " jobs).\n";

                // check that all jobs are done
                BOOST_CHECK_EQUAL(pool.n_workers(), thread::n_threads());
                BOOST_CHECK_EQUAL(pool.n_tasks(), 0);

                BOOST_CHECK_EQUAL(tasks_done.size(), n_tasks);
                for (size_t j = 0; j < n_tasks; ++ j)
                {
                        BOOST_CHECK(std::find(tasks_done.begin(), tasks_done.end(), j + 1) != tasks_done.end());
                }
        }
}
