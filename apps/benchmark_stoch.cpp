#include "logger.h"
#include "math/abs.hpp"
#include "text/table.h"
#include "optim/stoch.h"
#include "text/cmdline.h"
#include "math/clamp.hpp"
#include "math/random.hpp"
#include "math/epsilon.hpp"
#include "math/numeric.hpp"
#include "benchmark_optimizers.h"
#include <map>
#include <tuple>

using namespace nano;

template <typename tostats>
void check_function(
        const function_t& function, const size_t trials, const size_t epochs, const size_t epoch_size, const scalar_t epsilon,
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
        const auto optimizers = enum_values<stoch_optimizer>();

        // per-problem statistics
        tostats stats;

        // evaluate all optimizers
        for (const auto optimizer : optimizers)
        {
                const auto params = stoch_params_t(epochs, epoch_size, optimizer);
                const auto op = [&] (const problem_t& problem, const vector_t& x0)
                {
                        auto state = minimize(params, problem, x0);
                        if (state.converged(epsilon))
                        {
                                state.m_status = opt_status::converged;
                        }
                        return state;
                };

                const auto name =
                        to_string(optimizer);

                benchmark::benchmark_function(function, x0s, op, name, stats, gstats);
        }

        // show per-problem statistics
        benchmark::show_table(function.name(), stats);
}

int main(int argc, const char* argv[])
{
        // parse the command line
        cmdline_t cmdline("benchmark stochastic optimizers");
        cmdline.add("", "min-dims",     "minimum number of dimensions for each test function (if feasible)", "100");
        cmdline.add("", "max-dims",     "maximum number of dimensions for each test function (if feasible)", "1000");
        cmdline.add("", "trials",       "number of random trials for each test function", "100");
        cmdline.add("", "epochs",       "optimization: number of epochs", "100");
        cmdline.add("", "epoch-size",   "optimization: number of iterations per epoch", "100");
        cmdline.add("", "epsilon",      "convergence criteria", std::sqrt(nano::epsilon3<scalar_t>()));
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

