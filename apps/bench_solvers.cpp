#include "solver.h"
#include "core/stats.h"
#include "core/table.h"
#include "core/tpool.h"
#include "core/cmdline.h"
#include "core/numeric.h"
#include "core/algorithm.h"
#include "solvers/lsearch.h"
#include <iostream>

using namespace nano;

struct solver_stat_t
{
        void update(const solver_state_t& state)
        {
                m_crits(state.convergence_criteria());
                m_fails(state.m_status != solver_state_t::status::converged ? 1 : 0);
                m_iters(static_cast<scalar_t>(state.m_iterations));
                m_errors(state.m_status == solver_state_t::status::failed ? 1 : 0);
                m_maxits(state.m_status == solver_state_t::status::max_iters ? 1 : 0);
                m_fcalls(static_cast<scalar_t>(state.m_fcalls));
                m_gcalls(static_cast<scalar_t>(state.m_gcalls));
        }

        stats_t         m_crits;      ///< convergence criteria
        stats_t         m_fails;      ///< #convergence failures
        stats_t         m_iters;      ///< #optimization iterations
        stats_t         m_errors;     ///< #internal errors (e.g. line-search failed)
        stats_t         m_maxits;     ///< #maximum iterations reached
        stats_t         m_fcalls;     ///< #function value calls
        stats_t         m_gcalls;     ///< #gradient calls
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
                << "#iters"
                << "#errors"
                << "#maxits"
                << "#fcalls"
                << "#gcalls";
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
                        << static_cast<size_t>(stat.m_iters.avg())
                        << static_cast<size_t>(stat.m_errors.sum1())
                        << static_cast<size_t>(stat.m_maxits.sum1())
                        << static_cast<size_t>(stat.m_fcalls.avg())
                        << static_cast<size_t>(stat.m_gcalls.avg());
                }
        }

        table.sort(nano::make_less_from_string<scalar_t>(), {3, 8, 2});
        std::cout << table;
}

static auto trim(const json_t& json)
{
        string_t config = json.dump();
        config = nano::replace(config, R"("inits":")" + join(enum_values<lsearch_t::initializer>()) + "\"", "");
        config = nano::replace(config, R"("strats":")" + join(enum_values<lsearch_t::strategy>()) + "\"", "");
        config = nano::replace(config, ",,", ",");
        config = nano::replace(config, "\"", "");
        config = nano::replace(config, ",}", "");
        config = nano::replace(config, "}", "");
        config = nano::replace(config, "{", "");

        return config;
}

static void check_solver(const function_t& function, const rsolver_t& solver, const string_t& id,
        const std::vector<vector_t>& x0s, const size_t iterations, const scalar_t epsilon,
        solver_config_stats_t& fstats, solver_config_stats_t& gstats)
{
        json_t json;
        solver->to_json(json);
        const auto config = trim(json);

        std::vector<solver_state_t> states(x0s.size());
        nano::loopi(x0s.size(), [&] (const size_t i)
        {
                states[i] = solver->minimize(iterations, epsilon, function, x0s[i]);
        });

        for (const auto& state : states)
        {
                fstats[std::make_pair(id, config)].update(state);
                gstats[std::make_pair(id, config)].update(state);
        }
}

static void check_function(const function_t& function, const std::vector<std::pair<string_t, rsolver_t>>& id_solvers,
        const size_t trials, const size_t iterations, const scalar_t epsilon, const bool is_tuning,
        solver_config_stats_t& gstats)
{
        // generate fixed random trials
        std::vector<vector_t> x0s(trials);
        std::generate(x0s.begin(), x0s.end(), [&] () { return vector_t::Random(function.size()); });

        // per-problem statistics
        solver_config_stats_t fstats;

        // evaluate all possible combinations (solver & line-search)
        for (const auto& id_solver : id_solvers)
        {
                const auto& id = id_solver.first;
                const auto& solver = id_solver.second;

                if (is_tuning)
                {
                        const auto tuner = solver->tuner();
                        for (const auto& json : tuner.get(tuner.n_configs()))
                        {
                                solver->from_json(json);
                                check_solver(function, solver, id, x0s, iterations, epsilon, fstats, gstats);
                        }
                }
                else
                {
                        check_solver(function, solver, id, x0s, iterations, epsilon, fstats, gstats);
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
        cmdline.add("", "epsilon",      "convergence criteria", epsilon2<scalar_t>());
        cmdline.add("", "convex",       "use only convex test functions");
        cmdline.add("", "tune",         "tune the selected solvers");
        cmdline.add("", "c1",           "use this c1 value (see Armijo-Goldstein line-search step condition)");
        cmdline.add("", "c2",           "use this c2 value (see Wolfe line-search step condition)");
        cmdline.add("", "ls-init",      "use this line-search initialization (" + join(enum_values<lsearch_t::initializer>()) + ")");
        cmdline.add("", "ls-strat",     "use this line-search strategy (" + join(enum_values<lsearch_t::strategy>()) + ")");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto min_dims = cmdline.get<tensor_size_t>("min-dims");
        const auto max_dims = cmdline.get<tensor_size_t>("max-dims");
        const auto trials = cmdline.get<size_t>("trials");
        const auto iterations = cmdline.get<size_t>("iterations");
        const auto epsilon = cmdline.get<scalar_t>("epsilon");
        const auto is_convex = cmdline.has("convex");
        const auto is_tuning = cmdline.has("tune");

        const auto fregex = std::regex(cmdline.get<string_t>("functions"));
        const auto sregex = std::regex(cmdline.get<string_t>("solvers"));

        std::vector<std::pair<string_t, rsolver_t>> solvers;
        for (const auto& id : get_solvers().ids(sregex))
        {
                auto solver = get_solver(id);
                if (cmdline.has("c1"))
                {
                        solver->from_json(nano::to_json("c1", cmdline.get<scalar_t>("c1")));
                }
                if (cmdline.has("c2"))
                {
                        solver->from_json(nano::to_json("c2", cmdline.get<scalar_t>("c2")));
                }
                if (cmdline.has("ls-init"))
                {
                        solver->from_json(nano::to_json("init", cmdline.get<lsearch_t::initializer>("ls-init")));
                }
                if (cmdline.has("ls-strat"))
                {
                        solver->from_json(nano::to_json("strat", cmdline.get<lsearch_t::strategy>("ls-strat")));
                }

                solvers.emplace_back(id, std::move(solver));
        }

        solver_config_stats_t gstats;
        for (const auto& function : (is_convex ? get_convex_functions : get_functions)(min_dims, max_dims, fregex))
        {
                check_function(*function, solvers, trials, iterations, epsilon, is_tuning, gstats);
        }

        show_table("Solver", gstats);

        // OK
        return EXIT_SUCCESS;
}
