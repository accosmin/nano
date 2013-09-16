#include "ncv.h"
#include "core/thread.h"
#include "core/logger.h"
#include "core/random.hpp"

int main(int argc, char *argv[])
{
        using namespace ncv;

        worker_pool_t& pool = worker_pool_t::instance();

        const size_t n_tests = 8;
        const size_t n_max_jobs = pool.n_workers() * 16;

        // run multiple tests ...
        for (size_t t = 0; t < n_tests; t ++)
        {
                random_t<size_t> rnd(1, n_max_jobs);

                // ... enqueue jobs
                const size_t n_jobs = rnd();
                log_info() << "@pool [" << (t + 1) << "/" << n_tests
                                << "]: creating " << n_jobs << " jobs ...";

                for (size_t j = 0; j < n_jobs; j ++)
                {
                        pool.enqueue([=]()
                        {
                                const size_t sleep1 = random_t<size_t>(10, 100)();
                                std::this_thread::sleep_for(std::chrono::milliseconds(sleep1));

                                log_info() << "#job [" << (j + 1) << "/" << n_jobs << "@"
                                                << (t + 1) << "/" << n_tests << "] started ...";

                                const size_t sleep2 = random_t<size_t>(10, 500)();
                                std::this_thread::sleep_for(std::chrono::milliseconds(sleep2));

                                log_info() << "#job [" << (j + 1) << "/" << n_jobs << "@"
                                                << (t + 1) << "/" << n_tests << "] done.";
                        });
                }

                // ... wait for all jobs to finish
                log_info() << "@pool [" << (t + 1) << "/" << n_tests
                                << "]: waiting for " << pool.n_jobs() << " jobs ...";
                pool.wait();
                log_info() << "@pool [" << (t + 1) << "/" << n_tests
                                << "]: waiting done (enqueued " << pool.n_jobs() << " jobs).";
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
