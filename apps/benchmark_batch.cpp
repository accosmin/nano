#include "math/abs.hpp"
#include "text/table.h"
#include "math/clamp.hpp"
#include "text/cmdline.h"
#include "optim/batch.hpp"
#include "math/random.hpp"
#include "cortex/logger.h"
#include "math/epsilon.hpp"
#include "math/numeric.hpp"
#include "math/epsilon.hpp"
#include "text/algorithm.h"
#include "benchmark_optimizers.h"
#include <map>
#include <tuple>

using namespace nano;

template <typename tostats>
static void check_function(
        const function_t& function, const size_t trials, const size_t iterations, const scalar_t epsilon,
        tostats& gstats)
{
        const auto dims = function.problem().size();

        random_t<scalar_t> rgen(scalar_t(-1), scalar_t(+1));

        // generate fixed random trials
        std::vector<vector_t> x0s(trials);
        for (auto& x0 : x0s)
        {
                x0.resize(dims);
                rgen(x0.data(), x0.data() + x0.size());
        }

        // optimizers to try
        const auto optimizers = enum_values<batch_optimizer>();

        // line search initialization methods to try
        const auto ls_initializers = enum_values<ls_initializer>();

        // line search strategies to try
        const auto ls_strategies = enum_values<ls_strategy>();

        // per-problem statistics
        tostats stats;

        // evaluate all possible combinations
        for (batch_optimizer optimizer : optimizers)
                for (ls_initializer ls_init : ls_initializers)
                        for (ls_strategy ls_strat : ls_strategies)
        {
                const auto op = [&] (const problem_t& problem, const vector_t& x0)
                {
                        return minimize(problem, nullptr, x0, optimizer, iterations, epsilon, ls_init, ls_strat);
                };

                const auto name =
                        to_string(optimizer) + "[" +
                        to_string(ls_init) + "][" +
                        to_string(ls_strat) + "]";

                benchmark::benchmark_function(function, x0s, op, name, stats, gstats);
        }

        // show per-problem statistics
        benchmark::show_table(function.name(), stats);
}

int main(int argc, const char* argv[])
{
        using namespace nano;

        // parse the command line
        cmdline_t cmdline("benchmark batch optimizers");
        cmdline.add("", "min-dims",     "minimum number of dimensions for each test function (if feasible)", "100");
        cmdline.add("", "max-dims",     "maximum number of dimensions for each test function (if feasible)", "1000");
        cmdline.add("", "trials",       "number of random trials for each test function", "100");
        cmdline.add("", "iterations",   "maximum number of iterations", "8000");
        cmdline.add("", "epsilon",      "convergence criteria", nano::epsilon3<scalar_t>());
        cmdline.add("", "convex",       "use only convex test functions");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto min_dims = cmdline.get<tensor_size_t>("min-dims");
        const auto max_dims = cmdline.get<tensor_size_t>("max-dims");
        const auto trials = cmdline.get<size_t>("trials");
        const auto iterations = cmdline.get<size_t>("iterations");
        const auto epsilon = cmdline.get<scalar_t>("epsilon");
        const auto is_convex = cmdline.has("convex");

        std::map<std::string, benchmark::optimizer_stat_t> gstats;

        const auto functions = (is_convex ? make_convex_functions : make_functions)(min_dims, max_dims);
        foreach_test_function(functions, [&] (const function_t& function)
        {
                check_function(function, trials, iterations, epsilon, gstats);
        });

        // show global statistics
        benchmark::show_table(std::string(), gstats);

        // show per-optimizer statistics
        for (const batch_optimizer optimizer : enum_values<batch_optimizer>())
        {
                const auto name = to_string(optimizer) + "[";

                std::map<std::string, benchmark::optimizer_stat_t> stats;
                for (const auto& gstat : gstats)
                {
                        if (starts_with(gstat.first, name))
                        {
                                stats[gstat.first] = gstat.second;
                        }
                }

                benchmark::show_table(std::string(), stats);
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}

