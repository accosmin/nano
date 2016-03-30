#include "math/abs.hpp"
#include "text/table.h"
#include "text/cmdline.h"
#include "math/stoch.hpp"
#include "math/clamp.hpp"
#include "cortex/logger.h"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "cortex/optimizer.h"
#include "text/from_string.hpp"
#include "math/funcs/foreach.hpp"
#include "benchmark_optimizers.h"
#include <map>
#include <tuple>

namespace
{
        using namespace nano;

        template
        <
                typename tscalar,
                typename tostats,
                typename tsize = typename nano::function_t<tscalar>::tsize,
                typename tvector = typename nano::function_t<tscalar>::tvector,
                typename tproblem = typename nano::function_t<tscalar>::tproblem
        >
        void check_function(const nano::function_t<tscalar>& function,
                const size_t trials, const size_t epochs, const size_t epoch_size,
                tostats& gstats)
        {
                const auto dims = function.problem().size();

                nano::random_t<tscalar> rgen(tscalar(-1), tscalar(+1));

                // generate fixed random trials
                std::vector<tvector> x0s(trials);
                for (auto& x0 : x0s)
                {
                        x0.resize(dims);
                        rgen(x0.data(), x0.data() + x0.size());
                }

                // optimizers to try
                const auto optimizers =
                {
                        nano::stoch_optimizer::SG,
                        nano::stoch_optimizer::SGM,
                        nano::stoch_optimizer::AG,
                        nano::stoch_optimizer::AGFR,
                        nano::stoch_optimizer::AGGR,
                        nano::stoch_optimizer::ADAGRAD,
                        nano::stoch_optimizer::ADADELTA,
                        nano::stoch_optimizer::ADAM
                };

                // per-problem statistics
                tostats stats;

                // evaluate all optimizers
                for (const auto optimizer : optimizers)
                {
                        const auto op = [&] (const tproblem& problem, const tvector& x0)
                        {
                                return nano::minimize(problem, nullptr, nullptr, x0, optimizer, epochs, epoch_size);
                        };

                        const auto name =
                                nano::to_string(optimizer);

                        benchmark::benchmark_function(function, x0s, op, name, {1e-6, 1e-5, 1e-4, 1e-3}, stats, gstats);
                }

                // show per-problem statistics
                benchmark::show_table(function.name(), stats);
        }
}

int main(int argc, const char* argv[])
{
        using namespace nano;

        // parse the command line
        nano::cmdline_t cmdline("benchmark stochastic optimizers");
        cmdline.add("", "min-dims",     "minimum number of dimensions for each test function (if feasible)", "100");
        cmdline.add("", "max-dims",     "maximum number of dimensions for each test function (if feasible)", "1000");
        cmdline.add("", "trials",       "number of random trials for each test function", "100");
        cmdline.add("", "epochs",       "optimization: number of epochs", "400");
        cmdline.add("", "epoch-size",   "optimization: number of iterations per epoch", "200");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto min_dims = cmdline.get<tensor_size_t>("min-dims");
        const auto max_dims = cmdline.get<tensor_size_t>("max-dims");
        const auto trials = cmdline.get<size_t>("trials");
        const auto epochs = cmdline.get<size_t>("epochs");
        const auto epoch_size = cmdline.get<size_t>("epoch-size");

        std::map<std::string, benchmark::optimizer_stat_t> gstats;

        nano::foreach_test_function<scalar_t, nano::test_type::all>(min_dims, max_dims,
                [&] (const nano::function_t<scalar_t>& function)
        {
                check_function(function, trials, epochs, epoch_size, gstats);
        });

        // show global statistics
        benchmark::show_table(std::string(), gstats);

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}

