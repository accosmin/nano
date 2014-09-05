#include "nanocv.h"

int main(int argc, char *argv[])
{
        ncv::init();

        using namespace ncv;

        const scalar_t avg = 0.03;
        const scalar_t var = 0.005;
        const size_t size = 32;

        ncv::stats_t<scalar_t> stats;
        ncv::random_t<scalar_t> rgen(-var, +var);

        for (size_t i = 0; i < size; i ++)
        {
                stats(avg + rgen());
        }

        log_info() << "size = " << stats.count();
        log_info() << "mean = " << stats.avg();
        log_info() << "stdev = " << stats.stdev();
        log_info() << "range = [" << stats.min() << ", " << stats.max() << "]";

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
