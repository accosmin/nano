#include "nanocv/text.h"
#include "nanocv/timer.h"
#include "nanocv/logger.h"
#include "nanocv/tensor.h"
#include "nanocv/math/abs.hpp"
#include "nanocv/math/stats.hpp"
#include "nanocv/thread/parallel.hpp"

template
<
        typename tvector
>
static bool check(const tvector& a, const tvector& b, const char* error_message)
{
        typedef typename tvector::Scalar scalar_t;

        const auto eps = std::numeric_limits<scalar_t>::epsilon();

        ncv::stats_t<scalar_t> stats;
        for (auto i = 0; i < a.size(); i ++)
        {
                stats(ncv::math::abs(b(i) - a(i)));
        }

        if (stats.max() > eps)
        {
                ncv::log_error() << error_message << " (diff = [" << stats.min() << ", " << stats.max()
                                 << "] ~ " << stats.avg() << ")";
                return false;
        }

        else
        {
                return true;
        }
}

int main(int, char* [])
{
        using namespace ncv;

        const size_t tests = 1000;
        const size_t minsize = 1024;
        const size_t maxsize = 1024 * 1024;

        thread_pool_t pool;

        // try various data sizes
        for (size_t size = minsize; size <= maxsize; size *= 2)
        {
                ncv::stats_t<double, size_t> scpu_stats;        // cpu processing (single core)                        
                ncv::stats_t<double, size_t> mcpu_stats;        // cpu processing (multiple cores)

                ncv::tensor::vector_types_t<double>::tvector a; a.resize(size);
                ncv::tensor::vector_types_t<double>::tvector b; b.resize(size);

                // run multiple tests
                for (size_t test = 0; test < tests; test ++)
                {
                        ncv::timer_t timer;
                        
                        a.setRandom();
                        b.setRandom();

                        // CPU - single-threaded
                        timer.start();
                        {
                                for (size_t i = 0; i < size; i ++)
                                {
                                        b(i) = a(i);
                                }
                        }
                        scpu_stats(timer.microseconds());

                        // CPU - multi-threaded
                        timer.start();
                        {
                                ncv::thread_loopi(size, pool, [&] (size_t i)
                                {
                                        b(i) = a(i);
                                });
                        }
                        mcpu_stats(timer.microseconds());
                }

                const size_t time_scpu = static_cast<size_t>(0.5 + scpu_stats.sum() / 1000);
                const size_t time_mcpu = static_cast<size_t>(0.5 + mcpu_stats.sum() / 1000);

                // results
                log_info() << "SIZE [" << text::resize(text::to_string(size / 1024), 4, align::right) << "K] x "
                            << "TIMES [" << text::resize(text::to_string(tests), 0, align::right) << "]: "
                            << "singCPU= " << text::resize(text::to_string(time_scpu), 8, align::right) << "ms, "
                            << "multCPU= " << text::resize(text::to_string(time_mcpu), 8, align::right) << "ms";
        }

        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
