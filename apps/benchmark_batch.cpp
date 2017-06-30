#include "math/random.h"
#include "text/cmdline.h"
#include "math/epsilon.h"
#include "functions/test.h"
#include "solver_batch.h"
#include "benchmark_optimizers.h"

using namespace nano;

template <typename tostats>
static void check_function(const function_t& function,
        const size_t trials, const size_t iterations, const scalar_t epsilon, tostats& gstats)
{
        auto rgen = make_rng(scalar_t(-1), scalar_t(+1));

        // generate fixed random trials
        std::vector<vector_t> x0s(trials);
        for (auto& x0 : x0s)
        {
                x0.resize(function.size());
                rgen(x0.data(), x0.data() + x0.size());
        }

        // per-problem statistics
        tostats stats;

        // evaluate all possible combinations (optimizer & line-search)
        for (const auto id : get_batch_solvers().ids())
                for (const ls_initializer ls_init : enum_values<ls_initializer>())
                        for (const ls_strategy ls_strat : enum_values<ls_strategy>())
        {
                const auto solver = get_batch_solvers().get(id, to_params("ls_init", ls_init, "ls_strat", ls_strat));
                const auto params = batch_params_t(iterations, epsilon);
                const auto name = id + "[" + to_string(ls_init) + "][" + to_string(ls_strat) + "]";

                const auto op = [&] (const function_t& f, const vector_t& x0)
                {
                        return solver->minimize(params, f, x0);
                };

                benchmark::benchmark_function(op, function, x0s, name, stats, gstats);
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
        cmdline.add("", "iterations",   "maximum number of iterations", "1000");
        cmdline.add("", "epsilon",      "convergence criteria", 1e-6);
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
        const auto ids = get_batch_solvers().ids();
        for (const auto optimizer : ids)
        {
                const auto name = optimizer + "[";

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
        return EXIT_SUCCESS;
}
