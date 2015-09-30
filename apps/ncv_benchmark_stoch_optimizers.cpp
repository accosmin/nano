#include "core/table.h"
#include "core/timer.h"
#include "math/abs.hpp"
#include "core/logger.h"
#include "text/align.hpp"
#include "math/clamp.hpp"
#include "math/stats.hpp"
#include "math/random.hpp"
#include "core/minimize.h"
#include "math/numeric.hpp"
#include "thread/loopi.hpp"
#include "math/tune_fixed.hpp"
#include "text/from_string.hpp"
#include "core/table_row_comp.h"
#include "func/make_functions.h"
#include <map>
#include <tuple>

using namespace ncv;

namespace
{
        struct optimizer_stat_t
        {
                math::stats_t<scalar_t>       m_times;        ///< optimization time (microseconds)
                math::stats_t<scalar_t>       m_crits;        ///< convergence criteria
                math::stats_t<scalar_t>       m_fail0s;       ///< #convergence failures
                math::stats_t<scalar_t>       m_fail1s;       ///< #convergence failures
                math::stats_t<scalar_t>       m_fail2s;       ///< #convergence failures
                math::stats_t<scalar_t>       m_fail3s;       ///< #convergence failures
                math::stats_t<scalar_t>       m_speeds;       ///< convergence speed (actually the average decrease ratio of the convergence criteria)
        };

        math::stats_t<scalar_t> make_stats(const scalars_t& values, const scalars_t& flags)
        {
                assert(values.size() == flags.size());

                math::stats_t<scalar_t> stats;
                for (size_t i = 0; i < values.size(); i ++)
                {
                        if (flags[i] >= 0.0)
                        {
                                stats(values[i]);
                        }
                }
                return stats;
        }

        void show_table(const string_t& name, const std::map<string_t, optimizer_stat_t>& ostats)
        {
                // show global statistics
                table_t table(text::align(name.empty() ? string_t("optimizer") : (name + " optimizer"), 32));
                table.header() << "time [us]"
                               << "|grad|/|fval|"
                               << "#>1e-3"
                               << "#>1e-4"
                               << "#>1e-5"
                               << "#>1e-6"
                               << "speed";

                for (const auto& it : ostats)
                {
                        const auto& name = it.first;
                        const auto& stat = it.second;

                        table.append(name) << stat.m_times.avg()
                                           << stat.m_crits.avg()
                                           << static_cast<size_t>(stat.m_fail3s.sum())
                                           << static_cast<size_t>(stat.m_fail2s.sum())
                                           << static_cast<size_t>(stat.m_fail1s.sum())
                                           << static_cast<size_t>(stat.m_fail0s.sum())
                                           << stat.m_speeds.avg();
                }

                table.sort(ncv::make_table_row_ascending_comp<scalar_t>(indices_t({2, 3, 4, 0})));
                table.print(std::cout);
        }

        static decltype(auto) tune(const opt_problem_t problem, const opt_vector_t x0,
                min::stoch_optimizer optimizer, opt_size_t epoch_size)
        {
                const std::vector<opt_scalar_t> alpha0s =
                {
                        1e+0, 1e-1, 1e-2, 1e-3
                };
                const std::vector<opt_scalar_t> decays =
                {
                        0.50, 0.75, 1.00
                };

                const auto op = [&] (const opt_scalar_t alpha0, const opt_scalar_t decay)
                {
                        const auto state = ncv::minimize(problem, nullptr, x0, optimizer, 1, epoch_size, alpha0, decay);
                        return state.f;
                };

                return math::tune_fixed(op, alpha0s, decays);
        }

        template <typename tostats>
        void check_function(const function_t& func, tostats& ostats)
        {
                const auto epochs = opt_size_t(128);
                const auto epoch_size = opt_size_t(64);
                const auto trials = size_t(1024);

                const auto dims = func.problem().size();

                math::random_t<opt_scalar_t> rgen(-1.0, +1.0);

                // generate fixed random trials
                std::vector<opt_vector_t> x0s(trials);
                for (auto& x0 : x0s)
                {
                        x0.resize(dims);
                        rgen(x0.data(), x0.data() + x0.size());
                }

                // optimizers to try
                const auto optimizers =
                {
                        min::stoch_optimizer::SG,
                        min::stoch_optimizer::SGA,
                        min::stoch_optimizer::SIA,
                        min::stoch_optimizer::AG,
                        min::stoch_optimizer::AGGR,
                        min::stoch_optimizer::ADAGRAD,
                        min::stoch_optimizer::ADADELTA
                };

                thread::pool_t pool;

                // per-problem statistics
                tostats stats;

                // evaluate all optimizers
                for (const auto optimizer : optimizers)
                {
                        scalars_t times(trials);
                        scalars_t crits(trials);
                        scalars_t fail0s(trials);
                        scalars_t fail1s(trials);
                        scalars_t fail2s(trials);
                        scalars_t fail3s(trials);
                        scalars_t speeds(trials);

                        thread::loopi(trials, pool, [&] (size_t t)
                        {
                                const auto& x0 = x0s[t];

                                const auto problem = func.problem();
                                const auto state0 = opt_state_t(problem, x0);
                                const auto g0 = state0.convergence_criteria();

                                // optimize
                                const ncv::timer_t timer;

                                opt_scalar_t alpha0, decay, ftune;
                                std::tie(ftune, alpha0, decay) = tune(problem, x0, optimizer, epoch_size);

                                const auto state = ncv::minimize(
                                        problem, nullptr, x0, optimizer, epochs, epoch_size, alpha0, decay);

                                const auto g = state.convergence_criteria();
                                const auto speed = std::pow(g / g0, 1.0 / (1.0 + state.m_iterations));

                                // ignore out-of-domain solutions
                                if (func.is_valid(state.x))
                                {
                                        // update stats
                                        times[t] = timer.microseconds();
                                        crits[t] = g;
                                        fail0s[t] = !state.converged(1e-6) ? 1.0 : 0.0;
                                        fail1s[t] = !state.converged(1e-5) ? 1.0 : 0.0;
                                        fail2s[t] = !state.converged(1e-4) ? 1.0 : 0.0;
                                        fail3s[t] = !state.converged(1e-3) ? 1.0 : 0.0;
                                        speeds[t] = speed;
                                }
                                else
                                {
                                        // skip this from statistics!
                                        times[t] = -1.0;
                                }
                        });

                        // update per-problem statistics
                        const string_t name =
                                text::to_string(optimizer);

                        optimizer_stat_t& stat = stats[name];
                        stat.m_times(make_stats(times, times));
                        stat.m_crits(make_stats(crits, times));
                        stat.m_fail0s(make_stats(fail0s, times));
                        stat.m_fail1s(make_stats(fail1s, times));
                        stat.m_fail2s(make_stats(fail2s, times));
                        stat.m_fail3s(make_stats(fail3s, times));
                        stat.m_speeds(make_stats(speeds, times));

                        // update global statistics
                        optimizer_stat_t& ostat = ostats[name];
                        ostat.m_times(stat.m_times);
                        ostat.m_crits(stat.m_crits);
                        ostat.m_fail0s(stat.m_fail0s);
                        ostat.m_fail1s(stat.m_fail1s);
                        ostat.m_fail2s(stat.m_fail2s);
                        ostat.m_fail3s(stat.m_fail3s);
                        ostat.m_speeds(stat.m_speeds);
                }

                show_table(func.name(), stats);
        }

        template <typename tstats>
        void check_function(const functions_t& funcs, tstats& ostats)
        {
                for (const auto& func : funcs)
                {
                        check_function(*func, ostats);
                }
        }
}

int main(int, char* [])
{
        using namespace ncv;

        std::map<string_t, optimizer_stat_t> ostats;

        const auto funcs = ncv::make_all_test_functions(8);
        for (const auto& func : funcs)
        {
                check_function(*func, ostats);
        }

        // show global statistics
        show_table(string_t(), ostats);

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}

