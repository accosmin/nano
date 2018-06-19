#include "solver.h"
#include "math/stats.h"
#include "text/table.h"
#include "text/cmdline.h"
#include "math/epsilon.h"
#include "text/algorithm.h"
#include "solvers/lsearch.h"
#include <iostream>

using namespace nano;

struct solver_stat_t
{
        stats_t<scalar_t> m_crits;      ///< convergence criteria
        stats_t<scalar_t> m_fails;      ///< #convergence failures
        stats_t<scalar_t> m_fcalls;     ///< #function value calls
        stats_t<scalar_t> m_gcalls;     ///< #gradient calls
        stats_t<scalar_t> m_speeds;     ///< #convergence speeds
};

static void show_table(const string_t& table_name, const std::map<string_t, solver_stat_t>& ostats)
{
        assert(!ostats.empty());

        // show global statistics
        table_t table;
        table.header()
                << nano::align(table_name.empty() ? "solver" : table_name, 42)
                << "gnorm"
                << "#fails"
                << "#fcalls"
                << "#gcalls"
                << "speed";
        table.delim();

        for (const auto& it : ostats)
        {
                const auto& name = it.first;
                const auto& stat = it.second;

                if (stat.m_fcalls)
                {
                        table.append()
                        << name
                        << stat.m_crits.avg()
                        << static_cast<size_t>(stat.m_fails.sum1())
                        << static_cast<size_t>(stat.m_fcalls.avg())
                        << static_cast<size_t>(stat.m_gcalls.avg())
                        << stat.m_speeds.avg();
                }
        }

        table.sort(nano::make_less_from_string<scalar_t>(), {1});
        std::cout << table;
}

template <typename tsolver, typename tostats>
static void benchmark_function(const tsolver& solver,
        const function_t& function, const vector_t& x0, const string_t& name,
        tostats& stats, tostats& gstats)
{
        solver_state_t state0(function.size());
        state0.update(function, x0);
        const auto g0 = state0.convergence_criteria();

        // optimize
        const auto old_fcalls = function.fcalls();
        const auto old_gcalls = function.gcalls();

        const auto state = solver();

        const auto fcalls = function.fcalls() - old_fcalls;
        const auto gcalls = function.gcalls() - old_gcalls;

        const auto g = state.convergence_criteria();
        const auto speed = std::pow(
                static_cast<double>(epsilon0<scalar_t>() + g) /
                static_cast<double>(epsilon0<scalar_t>() + g0),
                double(1) / double(gcalls));

        // ignore out-of-domain solutions
        if (state && function.is_valid(state.x))
        {
                // update per-function statistics
                solver_stat_t& stat = stats[name];
                stat.m_crits(g);
                stat.m_fails(state.m_status != solver_state_t::status::converged ? 1 : 0);
                stat.m_fcalls(static_cast<scalar_t>(fcalls));
                stat.m_gcalls(static_cast<scalar_t>(gcalls));
                stat.m_speeds(static_cast<scalar_t>(speed));

                // update global statistics
                solver_stat_t& gstat = gstats[name];
                gstat.m_crits(g);
                gstat.m_fails(state.m_status != solver_state_t::status::converged ? 1 : 0);
                gstat.m_fcalls(static_cast<scalar_t>(fcalls));
                gstat.m_gcalls(static_cast<scalar_t>(gcalls));
                gstat.m_speeds(static_cast<scalar_t>(speed));
        }
}

template <typename tostats>
static void check_function(const function_t& function, const strings_t& solvers,
        const size_t trials, const size_t iterations, const scalar_t epsilon, const scalar_t c1, tostats& gstats)
{
        // generate fixed random trials
        std::vector<vector_t> x0s(trials);
        for (auto& x0 : x0s)
        {
                x0 = vector_t::Random(function.size());
        }

        // per-problem statistics
        tostats stats;

        // evaluate all possible combinations (solver & line-search)
        for (const auto& id : solvers)
                for (const auto ls_init : enum_values<lsearch_t::initializer>())
                        for (const auto ls_strat : enum_values<lsearch_t::strategy>())
        {
                const auto solver = get_solvers().get(id);
                solver->from_json(to_json("ls_init", ls_init, "ls_strat", ls_strat, "c1", c1));
                const auto name = id + "[" + to_string(ls_init) + "][" + to_string(ls_strat) + "]";

                for (const auto& x0 : x0s)
                {
                        benchmark_function(
                                [&] () { return solver->minimize(iterations, epsilon, function, x0); },
                                function, x0, name, stats, gstats);
                }
        }

        // show per-problem statistics
        show_table(function.name(), stats);
}

int main(int argc, const char* argv[])
{
        using namespace nano;

        // parse the command line
        cmdline_t cmdline("benchmark solvers");
        cmdline.add("", "solvers",      "use this regex to select the solvers to benchmark", ".+");
        cmdline.add("", "functions",    "use this regex to select the functions to benchmark", ".+");
        cmdline.add("", "min-dims",     "minimum number of dimensions for each test function (if feasible)", "100");
        cmdline.add("", "max-dims",     "maximum number of dimensions for each test function (if feasible)", "1000");
        cmdline.add("", "trials",       "number of random trials for each test function", "100");
        cmdline.add("", "iterations",   "maximum number of iterations", "1000");
        cmdline.add("", "epsilon",      "convergence criteria", 1e-6);
        cmdline.add("", "convex",       "use only convex test functions");
        cmdline.add("", "c1",           "sufficient decrease coefficient (Wolfe conditions)", 1e-4);

        cmdline.process(argc, argv);

        // check arguments and options
        const auto min_dims = cmdline.get<tensor_size_t>("min-dims");
        const auto max_dims = cmdline.get<tensor_size_t>("max-dims");
        const auto trials = cmdline.get<size_t>("trials");
        const auto iterations = cmdline.get<size_t>("iterations");
        const auto epsilon = cmdline.get<scalar_t>("epsilon");
        const auto is_convex = cmdline.has("convex");
        const auto c1 = cmdline.get<scalar_t>("c1");

        const auto solvers = get_solvers().ids(std::regex(cmdline.get<string_t>("solvers")));
        const auto functions = std::regex(cmdline.get<string_t>("functions"));

        std::map<std::string, solver_stat_t> gstats;

        for (const auto& function : (is_convex ? get_convex_functions : get_functions)(min_dims, max_dims, functions))
        {
                check_function(*function, solvers, trials, iterations, epsilon, c1, gstats);
        }

        // show global statistics
        show_table(std::string(), gstats);

        // show per-solver statistics
        for (const auto& solver : solvers)
        {
                const auto name = solver + "[";

                std::map<std::string, solver_stat_t> stats;
                for (const auto& gstat : gstats)
                {
                        if (starts_with(gstat.first, name))
                        {
                                stats[gstat.first] = gstat.second;
                        }
                }

                show_table(std::string(), stats);
        }

        // OK
        return EXIT_SUCCESS;
}
