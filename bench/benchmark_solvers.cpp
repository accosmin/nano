#include "solver.h"
#include "core/stats.h"
#include "core/table.h"
#include "core/cmdline.h"
#include "core/epsilon.h"
#include "core/algorithm.h"
#include "solvers/lsearch.h"
#include <iostream>

using namespace nano;

struct solver_stat_t
{
        void update(const function_t& function, const solver_state_t& state0, const solver_state_t& statex)
        {
                const auto g0 = state0.convergence_criteria();
                const auto gx = statex.convergence_criteria();

                const auto fcalls = function.fcalls();
                const auto gcalls = function.gcalls();

                const auto speed = std::pow(
                        static_cast<double>(epsilon0<scalar_t>() + gx) /
                        static_cast<double>(epsilon0<scalar_t>() + g0),
                        double(1) / double(gcalls));

                // ignore out-of-domain solutions
                if (statex && function.is_valid(statex.x))
                {
                        m_crits(gx);
                        m_fails(statex.m_status != solver_state_t::status::converged ? 1 : 0);
                        m_fcalls(static_cast<scalar_t>(fcalls));
                        m_gcalls(static_cast<scalar_t>(gcalls));
                        m_speeds(static_cast<scalar_t>(speed));
                }
        }

        stats_t<scalar_t> m_crits;      ///< convergence criteria
        stats_t<scalar_t> m_fails;      ///< #convergence failures
        stats_t<scalar_t> m_fcalls;     ///< #function value calls
        stats_t<scalar_t> m_gcalls;     ///< #gradient calls
        stats_t<scalar_t> m_speeds;     ///< #convergence speeds
};

using solver_config_stats_t = std::map<
        std::pair<string_t, string_t>,  ///< key: {solver id, solver config}
        solver_stat_t>;                 ///< value: solver statistics

static void show_table(const string_t& table_name, const solver_config_stats_t& stats)
{
        assert(!stats.empty());

        // show global statistics
        table_t table;
        table.header()
                << colspan(2) << table_name
                << "gnorm"
                << "#fails"
                << "#fcalls"
                << "#gcalls"
                << "speed";
        table.delim();

        for (const auto& it : stats)
        {
                const auto& id = it.first.first;
                const auto& config = it.first.second;
                const auto& stat = it.second;

                if (stat.m_fcalls)
                {
                        table.append()
                        << id
                        << config
                        << stat.m_crits.avg()
                        << static_cast<size_t>(stat.m_fails.sum1())
                        << static_cast<size_t>(stat.m_fcalls.avg())
                        << static_cast<size_t>(stat.m_gcalls.avg())
                        << stat.m_speeds.avg();
                }
        }

        table.sort(nano::make_less_from_string<scalar_t>(), {3, 5, 1});
        std::cout << table;
}

static auto trim(const json_t& json)
{
        string_t config = json.dump();
        config = nano::replace(config, "\"inits\":\"" + join(enum_values<lsearch_t::initializer>()) + "\"", "");
        config = nano::replace(config, "\"strats\":\"" + join(enum_values<lsearch_t::strategy>()) + "\"", "");
        config = nano::replace(config, ",,", ",");
        config = nano::replace(config, "\"", "");
        config = nano::replace(config, ",}", "");
        config = nano::replace(config, "}", "");
        config = nano::replace(config, "{", "");

        return config;
}

static auto check_solver(const function_t& function, const rsolver_t& solver, const string_t& id,
        const std::vector<vector_t>& x0s, const size_t iterations, const scalar_t epsilon)
{
        solver_config_stats_t stats;

        json_t json;
        solver->to_json(json);
        const auto config = trim(json);

        for (const auto& x0 : x0s)
        {
                const auto state0 = solver_state_t{function, x0};

                function.reset_calls();
                const auto statex = solver->minimize(iterations, epsilon, function, x0);

                stats[std::make_pair(id, config)].update(function, state0, statex);
        }

        return stats;
}

static void check_function(const function_t& function, const strings_t& solvers,
        const size_t trials, const size_t iterations, const scalar_t epsilon, const bool is_tuning,
        solver_config_stats_t& gstats)
{
        // generate fixed random trials
        std::vector<vector_t> x0s(trials);
        for (auto& x0 : x0s)
        {
                x0 = vector_t::Random(function.size());
        }

        // per-problem statistics
        solver_config_stats_t fstats;

        // evaluate all possible combinations (solver & line-search)
        for (const auto& id : solvers)
        {
                const auto solver = get_solvers().get(id);

                if (is_tuning)
                {
                        const auto tuner = solver->tuner();
                        for (const auto& json : tuner.get(tuner.n_configs()))
                        {
                                solver->from_json(json);

                                const auto stats = check_solver(function, solver, id, x0s, iterations, epsilon);
                                fstats.insert(stats.begin(), stats.end());
                                gstats.insert(stats.begin(), stats.end());
                        }
                }
                else
                {
                        const auto stats = check_solver(function, solver, id, x0s, iterations, epsilon);
                        fstats.insert(stats.begin(), stats.end());
                        gstats.insert(stats.begin(), stats.end());
                }
        }

        // show per-problem statistics
        show_table(function.name(), fstats);
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
        cmdline.add("", "tune",         "tune the selected solvers");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto min_dims = cmdline.get<tensor_size_t>("min-dims");
        const auto max_dims = cmdline.get<tensor_size_t>("max-dims");
        const auto trials = cmdline.get<size_t>("trials");
        const auto iterations = cmdline.get<size_t>("iterations");
        const auto epsilon = cmdline.get<scalar_t>("epsilon");
        const auto is_convex = cmdline.has("convex");
        const auto is_tuning = cmdline.has("tune");

        const auto solvers = get_solvers().ids(std::regex(cmdline.get<string_t>("solvers")));
        const auto functions = std::regex(cmdline.get<string_t>("functions"));

        solver_config_stats_t gstats;
        for (const auto& function : (is_convex ? get_convex_functions : get_functions)(min_dims, max_dims, functions))
        {
                check_function(*function, solvers, trials, iterations, epsilon, is_tuning, gstats);
        }

        show_table("Solver", gstats);

        // OK
        return EXIT_SUCCESS;
}
