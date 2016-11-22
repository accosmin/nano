#include "logger.h"
#include "text/cmdline.h"
#include "math/random.h"
#include "math/epsilon.h"
#include "math/numeric.h"
#include "stoch_optimizer.h"
#include "benchmark_optimizers.h"
#include <tuple>

using namespace nano;

template <typename tostats>
void check_function(
        const function_t& function, const size_t trials, const size_t epochs, const size_t epoch_size, const scalar_t epsilon,
        tostats& gstats)
{
        const auto dims = function.size();

        auto rgen = make_rng(scalar_t(-1), scalar_t(+1));

        // generate fixed random trials
        vectors_t x0s(trials);
        for (auto& x0 : x0s)
        {
                x0.resize(dims);
                rgen(x0.data(), x0.data() + x0.size());
        }

        // optimizers to try
        const auto ids = get_stoch_optimizers().ids();

        // per-problem statistics
        tostats stats;

        // evaluate all optimizers
        for (const auto id : ids)
        {
                const auto optimizer = get_stoch_optimizers().get(id);
                const auto params = stoch_params_t(epochs, epoch_size, epsilon);
                const auto op = [&] (const vector_t& x0)
                {
                        return optimizer->minimize(params, function, x0);
                };

                const auto name = id;

                benchmark::benchmark_function(function, x0s, op, name, stats, gstats);
        }

        // show per-problem statistics
        benchmark::show_table(function.name(), stats);
}

int main(int argc, const char* argv[])
{
        // parse the command line
        cmdline_t cmdline("benchmark stochastic optimizers");
        cmdline.add("", "min-dims",     "minimum number of dimensions for each test function (if feasible)", "10");
        cmdline.add("", "max-dims",     "maximum number of dimensions for each test function (if feasible)", "100");
        cmdline.add("", "trials",       "number of random trials for each test function", "100");
        cmdline.add("", "epochs",       "optimization: number of epochs", "1000");
        cmdline.add("", "epoch-size",   "optimization: number of iterations per epoch", "100");
        cmdline.add("", "epsilon",      "convergence criteria", nano::epsilon2<scalar_t>());
        cmdline.add("", "convex",       "use only convex test functions");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto min_dims = cmdline.get<tensor_size_t>("min-dims");
        const auto max_dims = cmdline.get<tensor_size_t>("max-dims");
        const auto trials = cmdline.get<size_t>("trials");
        const auto epochs = cmdline.get<size_t>("epochs");
        const auto epoch_size = cmdline.get<size_t>("epoch-size");
        const auto epsilon = cmdline.get<scalar_t>("epsilon");
        const auto is_convex = cmdline.has("convex");

        std::map<std::string, benchmark::optimizer_stat_t> gstats;

        const auto functions = (is_convex ? make_convex_functions : make_functions)(min_dims, max_dims);
        foreach_test_function(functions, [&] (const function_t& function)
        {
                check_function(function, trials, epochs, epoch_size, epsilon, gstats);
        });

        // show global statistics
        benchmark::show_table(std::string(), gstats);

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}

