#include "tensor.h"
#include "math/abs.hpp"
#include "text/table.h"
#include "optim/test.h"
#include "text/align.hpp"
#include "math/clamp.hpp"
#include "math/stats.hpp"
#include "thread/loopi.hpp"

namespace benchmark
{
        using namespace nano;

        struct optimizer_stat_t
        {
                stats_t<scalar_t> m_crits;      ///< convergence criteria
                stats_t<scalar_t> m_fails;      ///< #convergence failures
                stats_t<scalar_t> m_iters;      ///< #iterations
                stats_t<scalar_t> m_fcalls;     ///< #function value calls
                stats_t<scalar_t> m_gcalls;     ///< #gradient calls
                stats_t<scalar_t> m_speeds;     ///< convergence speed (actually the average decrease ratio of the convergence criteria)
        };

        stats_t<scalar_t> make_stats(const scalars_t& values, const scalars_t& flags)
        {
                assert(values.size() == flags.size());

                nano::stats_t<scalar_t> stats;
                for (size_t i = 0; i < values.size(); i ++)
                {
                        if (flags[i] >= 0)
                        {
                                stats(values[i]);
                        }
                }
                return stats;
        }

        void show_table(const std::string& table_name, const std::map<std::string, optimizer_stat_t>& ostats)
        {
                assert(!ostats.empty());

                // show global statistics
                nano::table_t table(nano::align(table_name.empty() ? "optimizer" : table_name, 24));
                table.header() << "cost"
                               << "|g|/(1+|f|)"
                               << "#fails"
                               << "#iters"
                               << "#fcalls"
                               << "#gcalls"
                               << "speed";

                for (const auto& it : ostats)
                {
                        const auto& name = it.first;
                        const auto& stat = it.second;

                        table.append(name) << static_cast<size_t>(stat.m_fcalls.avg() + 2 * stat.m_gcalls.avg())
                                           << stat.m_crits.avg()
                                           << static_cast<size_t>(stat.m_fails.sum())
                                           << static_cast<size_t>(stat.m_iters.avg())
                                           << static_cast<size_t>(stat.m_fcalls.avg())
                                           << static_cast<size_t>(stat.m_gcalls.avg())
                                           << stat.m_speeds.avg();
                }

                table.sort<scalar_t>(table_t::sorting::asc, {2, 0});
                table.print(std::cout);
        }

        template
        <
                typename toptimizer,
                typename tostats
        >
        void benchmark_function(
                const function_t& func, const std::vector<vector_t>& x0s,
                const toptimizer& op, const std::string& name,
                tostats& stats, tostats& gstats)
        {
                const auto trials = x0s.size();

                scalars_t crits(trials);
                scalars_t iters(trials);
                scalars_t fails(trials);
                scalars_t fcalls(trials);
                scalars_t gcalls(trials);
                scalars_t speeds(trials);

                nano::loopi(trials, [&] (size_t t)
                {
                        const auto& x0 = x0s[t];

                        const auto problem = func.problem();
                        const auto state0 = state_t(problem, x0);
                        const auto g0 = state0.convergence_criteria();

                        // optimize
                        const auto state = op(problem, x0);

                        const auto g = state.convergence_criteria();
                        const auto cost = state.m_fcalls + 2 * state.m_gcalls;
                        const auto speed = std::pow(g / g0, 1 / (1 + static_cast<scalar_t>(cost)));

                        // ignore out-of-domain solutions
                        if (func.is_valid(state.x))
                        {
                                // update stats
                                crits[t] = g;
                                iters[t] = static_cast<scalar_t>(state.m_iterations);
                                fails[t] = (state.m_status != opt_status::converged) ? 1 : 0;
                                fcalls[t] = static_cast<scalar_t>(state.m_fcalls);
                                gcalls[t] = static_cast<scalar_t>(state.m_gcalls);
                                speeds[t] = speed;
                        }
                        else
                        {
                                // skip this from statistics!
                                crits[t] = -1;
                        }
                });

                // update per-problem statistics
                optimizer_stat_t& stat = stats[name];
                stat.m_crits(make_stats(crits, crits));
                stat.m_iters(make_stats(iters, crits));
                stat.m_fails(make_stats(fails, crits));
                stat.m_speeds(make_stats(speeds, crits));
                stat.m_fcalls(make_stats(fcalls, crits));
                stat.m_gcalls(make_stats(gcalls, crits));

                // update global statistics
                optimizer_stat_t& gstat = gstats[name];
                gstat.m_crits(stat.m_crits);
                gstat.m_iters(stat.m_iters);
                gstat.m_fails(stat.m_fails);
                gstat.m_speeds(stat.m_speeds);
                gstat.m_fcalls(stat.m_fcalls);
                gstat.m_gcalls(stat.m_gcalls);
        }
}
