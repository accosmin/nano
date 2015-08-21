#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_thread_pool"

#include <boost/test/unit_test.hpp>
#include "nanocv/logger.h"
#include "nanocv/thread/pool.h"
#include "nanocv/thread/thread.h"
#include "nanocv/math/random.hpp"

BOOST_AUTO_TEST_CASE(test_thread_pool)
{
        using namespace ncv;

        thread::pool_t pool;

        std::mutex mutex;

        // check that there is no job to do
        BOOST_CHECK_EQUAL(pool.n_workers(), ncv::n_threads());
        BOOST_CHECK_EQUAL(pool.n_tasks(), 0);

        const size_t n_tests = 8;
        const size_t n_max_jobs = pool.n_workers() * 16;

        // run multiple tests ...
        for (size_t t = 0; t < n_tests; t ++)
        {
                random_t<size_t> rnd(1, n_max_jobs);

                // ... enqueue jobs
                const size_t n_tasks = rnd();
                log_info() << "@pool [" << (t + 1) << "/" << n_tests
                           << "]: creating " << n_tasks << " jobs ...";

                std::vector<size_t> tasks_done;

                for (size_t j = 0; j < n_tasks; j ++)
                {
                        pool.enqueue([=, &mutex, &tasks_done]()
                        {
                                const size_t sleep1 = random_t<size_t>(5, 20)();
                                std::this_thread::sleep_for(std::chrono::milliseconds(sleep1));

                                {
                                        const std::lock_guard<std::mutex> lock(mutex);

                                        log_info() << "#job [" << (j + 1) << "/" << n_tasks << "@"
                                                   << (t + 1) << "/" << n_tests << "] started ...";
                                }

                                const size_t sleep2 = random_t<size_t>(5, 50)();
                                std::this_thread::sleep_for(std::chrono::milliseconds(sleep2));

                                {
                                        const std::lock_guard<std::mutex> lock(mutex);

                                        log_info() << "#job [" << (j + 1) << "/" << n_tasks << "@"
                                                   << (t + 1) << "/" << n_tests << "] done.";

                                        tasks_done.push_back(j + 1);
                                }
                        });
                }

                // ... wait for all jobs to finish
                log_info() << "@pool [" << (t + 1) << "/" << n_tests
                           << "]: waiting for " << pool.n_tasks() << " jobs ...";

                pool.wait();

                log_info() << "@pool [" << (t + 1) << "/" << n_tests
                                << "]: waiting done (enqueued " << pool.n_tasks() << " jobs).";

                // check that all jobs are done
                BOOST_CHECK_EQUAL(pool.n_workers(), ncv::n_threads());
                BOOST_CHECK_EQUAL(pool.n_tasks(), 0);

                BOOST_CHECK_EQUAL(tasks_done.size(), n_tasks);
                for (size_t j = 0; j < n_tasks; j ++)
                {
                        BOOST_CHECK(std::find(tasks_done.begin(), tasks_done.end(), j + 1) != tasks_done.end());
                }
        }
}
