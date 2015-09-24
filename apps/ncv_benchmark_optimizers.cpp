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
#include "math/epsilon.hpp"
#include "thread/loopi.hpp"
#include "text/from_string.hpp"
#include "core/table_row_comp.h"

#include "func/function_trid.h"
#include "func/function_beale.h"
#include "func/function_booth.h"
#include "func/function_cauchy.h"
#include "func/function_sphere.h"
#include "func/function_matyas.h"
#include "func/function_powell.h"
#include "func/function_mccormick.h"
#include "func/function_himmelblau.h"
#include "func/function_rosenbrock.h"
#include "func/function_3hump_camel.h"
#include "func/function_sum_squares.h"
#include "func/function_dixon_price.h"
#include "func/function_goldstein_price.h"
#include "func/function_rotated_ellipsoid.h"

#include <map>
#include <tuple>

using namespace ncv;

namespace
{
        struct optimizer_stat_t
        {
                stats_t<scalar_t>       m_time;
                stats_t<scalar_t>       m_crits;
                stats_t<scalar_t>       m_fails;
                stats_t<scalar_t>       m_iters;
                stats_t<scalar_t>       m_fvals;
                stats_t<scalar_t>       m_grads;
                stats_t<scalar_t>       m_speeds;
        };

        std::map<string_t, optimizer_stat_t> optimizer_stats;

        void check_problem(
                const string_t& problem_name,
                const opt_opsize_t& fn_size, const opt_opfval_t& fn_fval, const opt_opgrad_t& fn_grad,
                const std::vector<std::pair<vector_t, scalar_t>>&)
        {
                const size_t iterations = 1024;
                const scalar_t epsilon = 1e-6;
                const size_t trials = 1024;

                const size_t dims = fn_size();

                random_t<scalar_t> rgen(-1.0, +1.0);

                // generate fixed random trials
                vectors_t x0s(trials);
                for (auto& x0 : x0s)
                {
                        x0.resize(dims);
                        rgen(x0.data(), x0.data() + x0.size());
                }

                // optimizers to try
                const auto optimizers =
                {
                        min::batch_optimizer::GD,
                        min::batch_optimizer::CGD_CD,
                        min::batch_optimizer::CGD_DY,
                        min::batch_optimizer::CGD_FR,
                        min::batch_optimizer::CGD_HS,
                        min::batch_optimizer::CGD_LS,
                        min::batch_optimizer::CGD_DYCD,
                        min::batch_optimizer::CGD_DYHS,
                        min::batch_optimizer::CGD_PRP,
                        min::batch_optimizer::CGD_N,
                        min::batch_optimizer::LBFGS
                };

                const auto ls_initializers =
                {
                        min::ls_initializer::unit,
                        min::ls_initializer::quadratic,
                        min::ls_initializer::consistent
                };

                const auto ls_strategies =
                {
                        min::ls_strategy::backtrack_armijo,
                        min::ls_strategy::backtrack_wolfe,
                        min::ls_strategy::backtrack_strong_wolfe,
                        min::ls_strategy::interpolation,
                        min::ls_strategy::cg_descent
                };

                table_t table(text::align(problem_name, 32));
                table.header() << "cost"
                               << "time [us]"
                               << "|grad|/|fval|"
                               << "#fails"
                               << "#iters"
                               << "#fvals"
                               << "#grads"
                               << "speed";

                thread::pool_t pool;
                std::mutex mutex;

                for (min::batch_optimizer optimizer : optimizers)
                        for (min::ls_initializer ls_initializer : ls_initializers)
                                for (min::ls_strategy ls_strategy : ls_strategies)
                {
                        stats_t<scalar_t> times;
                        stats_t<scalar_t> crits;
                        stats_t<scalar_t> fails;
                        stats_t<scalar_t> iters;
                        stats_t<scalar_t> fvals;
                        stats_t<scalar_t> grads;
                        stats_t<scalar_t> speeds;

                        thread::loopi(trials, pool, [&] (size_t t)
                        {
                                const vector_t& x0 = x0s[t];

                                // check gradients
                                const opt_problem_t problem(fn_size, fn_fval, fn_grad);
                                if (problem.grad_accuracy(x0) > math::epsilon2<scalar_t>())
                                {
                                        const std::lock_guard<std::mutex> lock(mutex);

                                        log_error() << "invalid gradient for problem [" << problem_name << "]!";
                                }

                                // optimize
                                const ncv::timer_t timer;

                                const opt_state_t state = ncv::minimize(
                                        fn_size, fn_fval, fn_grad, nullptr, nullptr, nullptr,
                                        x0, optimizer, iterations, epsilon, ls_initializer, ls_strategy);

                                const auto crit = state.convergence_criteria();
                                const auto iter = state.n_iterations();

                                const opt_state_t state0(problem, x0);
                                const auto crit0 = state0.convergence_criteria();

                                const auto speed = std::pow(crit / crit0, 1.0 / (1.0 + iter));

                                // update stats
                                const std::lock_guard<std::mutex> lock(mutex);

                                times(timer.microseconds());
                                crits(crit);
                                iters(state.n_iterations());
                                fvals(state.n_fval_calls());
                                grads(state.n_grad_calls());
                                speeds(speed);

                                fails(!state.converged(epsilon) ? 1.0 : 0.0);
                        });

                        // update per-problem table
                        const string_t name =
                                text::to_string(optimizer) + "[" +
                                text::to_string(ls_initializer) + "][" +
                                text::to_string(ls_strategy) + "]";

                        table.append(name)
                                << static_cast<size_t>(fvals.sum() + 2 * grads.sum()) / trials
                                << times.avg()
                                << crits.avg()
                                << static_cast<size_t>(fails.sum())
                                << iters.avg()
                                << fvals.avg()
                                << grads.avg()
                                << speeds.avg();

                        // update global statistics
                        optimizer_stat_t& stat = optimizer_stats[name];
                        stat.m_time(times.avg());
                        stat.m_crits(crits.avg());
                        stat.m_fails(fails.sum());
                        stat.m_iters(iters.avg());
                        stat.m_fvals(fvals.avg());
                        stat.m_grads(grads.avg());
                        stat.m_speeds(speeds.avg());
                }

                // print stats
                table.sort(ncv::make_table_row_ascending_comp<scalar_t>(indices_t({3, 0})));
                table.print(std::cout);
        }

        void check_problems(const std::vector<ncv::function_t>& funcs)
        {
                for (const ncv::function_t& func : funcs)
                {
                        check_problem(func.m_name, func.m_opsize, func.m_opfval, func.m_opgrad, func.m_solutions);
                }
        }
}

int main(int, char* [])
{
        using namespace ncv;

//        check_problems(ncv::make_beale_funcs());
//        check_problems(ncv::make_booth_funcs());
//        check_problems(ncv::make_matyas_funcs());
//        check_problems(ncv::make_trid_funcs(32));
        check_problems(ncv::make_cauchy_funcs(32));
//        check_problems(ncv::make_sphere_funcs(32));
//        check_problems(ncv::make_powell_funcs(32));
//        check_problems(ncv::make_mccormick_funcs());
//        check_problems(ncv::make_himmelblau_funcs());
//        check_problems(ncv::make_rosenbrock_funcs(7));
//        check_problems(ncv::make_3hump_camel_funcs());
//        check_problems(ncv::make_dixon_price_funcs(32));
//        check_problems(ncv::make_sum_squares_funcs(32));
//        check_problems(ncv::make_goldstein_price_funcs());
//        check_problems(ncv::make_rotated_ellipsoid_funcs(32));

        // show global statistics
        table_t table(text::align("optimizer", 32));
        table.header() << "cost"
                       << "time [us]"
                       << "|grad|/|fval|"
                       << "#fails"
                       << "#iters"
                       << "#fvals"
                       << "#grads"
                       << "speed";

        for (const auto& it : optimizer_stats)
        {
                const string_t& name = it.first;
                const optimizer_stat_t& stat = it.second;

                table.append(name) << static_cast<int>(stat.m_fvals.sum() + 2 * stat.m_grads.sum())
                                   << stat.m_time.sum()
                                   << stat.m_crits.avg()
                                   << static_cast<int>(stat.m_fails.sum())
                                   << stat.m_iters.sum()
                                   << stat.m_fvals.sum()
                                   << stat.m_grads.sum()
                                   << stat.m_speeds.avg();
        }

        table.sort(ncv::make_table_row_ascending_comp<scalar_t>(indices_t({3, 0})));
        table.print(std::cout);

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}

