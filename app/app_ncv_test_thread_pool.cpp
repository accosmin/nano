#include "ncv_thread.h"
#include "ncv_logger.h"
#include "ncv_random.h"

int main(int argc, char *argv[])
{
        ncv::thread_impl::worker_pool& pool = ncv::thread_impl::worker_pool::instance();

        const ncv::size_t n_tests = 8;
        const ncv::size_t n_max_jobs = pool.n_threads() * 16;

        // Run multiple tests ...
        for (ncv::index_t t = 0; t < n_tests; t ++)
        {
                ncv::random<ncv::size_t> rnd(1, n_max_jobs);

                // ... enqueue jobs
                const ncv::size_t n_jobs = rnd();
                ncv::log_info() << "@pool [" << (t + 1) << "/" << n_tests
                                << "]: creating " << n_jobs << " jobs ...";

                for (ncv::index_t j = 0; j < n_jobs; j ++)
                {
                        pool.enqueue([=]()
                        {
                                const ncv::size_t sleep1 = ncv::random<ncv::size_t>(10, 100).operator()();
                                std::this_thread::sleep_for(std::chrono::milliseconds(sleep1));

                                ncv::log_info() << "#job [" << (j + 1) << "/" << n_jobs << "@"
                                                << (t + 1) << "/" << n_tests << "] started ...";

                                const ncv::size_t sleep2 = ncv::random<ncv::size_t>(10, 500).operator()();
                                std::this_thread::sleep_for(std::chrono::milliseconds(sleep2));

                                ncv::log_info() << "#job [" << (j + 1) << "/" << n_jobs << "@"
                                                << (t + 1) << "/" << n_tests << "] done.";
                        });
                }

                // ... wait for all jobs to finish
                ncv::log_info() << "@pool [" << (t + 1) << "/" << n_tests
                                << "]: waiting for " << pool.n_jobs() << " jobs ...";
                pool.wait();
                ncv::log_info() << "@pool [" << (t + 1) << "/" << n_tests
                                << "]: waiting done (enqueued " << pool.n_jobs() << " jobs).";
        }

        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
