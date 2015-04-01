#include "libnanocv/minimize.h"
#include "libnanocv/tabulator.h"
#include "libnanocv/math/abs.hpp"
#include "libnanocv/util/timer.h"
#include "libnanocv/math/math.hpp"
#include "libnanocv/util/logger.h"
#include "libnanocv/util/stats.hpp"
#include "libnanocv/math/clamp.hpp"
#include "libnanocv/util/random.hpp"
#include "libnanocv/math/epsilon.hpp"
#include "libnanocv/thread/parallel.hpp"

#include "libnanocv/functions/function_trid.h"
#include "libnanocv/functions/function_beale.h"
#include "libnanocv/functions/function_booth.h"
#include "libnanocv/functions/function_sphere.h"
#include "libnanocv/functions/function_matyas.h"
#include "libnanocv/functions/function_powell.h"
#include "libnanocv/functions/function_mccormick.h"
#include "libnanocv/functions/function_himmelblau.h"
#include "libnanocv/functions/function_rosenbrock.h"
#include "libnanocv/functions/function_3hump_camel.h"
#include "libnanocv/functions/function_sum_squares.h"
#include "libnanocv/functions/function_dixon_price.h"
#include "libnanocv/functions/function_goldstein_price.h"
#include "libnanocv/functions/function_rotated_ellipsoid.h"

#include <map>
#include <tuple>

using namespace ncv;

const size_t trials = 1024;

struct optimizer_stat_t
{
        stats_t<scalar_t>       m_time;
        stats_t<scalar_t>       m_crits;
        stats_t<scalar_t>       m_fails;
        stats_t<scalar_t>       m_iters;
        stats_t<scalar_t>       m_fvals;
        stats_t<scalar_t>       m_grads;
};

std::map<string_t, optimizer_stat_t> optimizer_stats;

static void check_problem(
        const string_t& problem_name,
        const opt_opsize_t& fn_size, const opt_opfval_t& fn_fval, const opt_opgrad_t& fn_grad,
        const std::vector<std::pair<vector_t, scalar_t>>&)
{
        const size_t iterations = 1024;
        const scalar_t epsilon = 1e-6;

        const size_t dims = fn_size();

        // generate fixed random trials
        vectors_t x0s;
        for (size_t t = 0; t < trials; t ++)
        {
                random_t<scalar_t> rgen(-1.0, +1.0);

                vector_t x0(dims);
                rgen(x0.data(), x0.data() + x0.size());

                x0s.push_back(x0);
        }

        // optimizers to try
        const auto optimizers =
        {
//                batch_optimizer::GD,
//                batch_optimizer::CGD_CD,
//                batch_optimizer::CGD_DY,
//                batch_optimizer::CGD_FR,
//                batch_optimizer::CGD_HS,
//                batch_optimizer::CGD_LS,
                batch_optimizer::CGD_PRP,
                batch_optimizer::CGD_N,
                batch_optimizer::CGD_DYCD,
                batch_optimizer::CGD_DYHS,
                batch_optimizer::LBFGS
        };

        const auto ls_initializers =
        {
                optimize::ls_initializer::unit,
                optimize::ls_initializer::quadratic,
                optimize::ls_initializer::consistent
        };

        const auto ls_strategies =
        {
//                optimize::ls_strategy::backtrack_armijo,
//                optimize::ls_strategy::backtrack_wolfe,
//                optimize::ls_strategy::backtrack_strong_wolfe,
//                optimize::ls_strategy::interpolation_bisection,
//                optimize::ls_strategy::interpolation_cubic,
                optimize::ls_strategy::cg_descent
        };

        tabulator_t table(text::resize(problem_name, 32));
        table.header() << "cost"
                       << "time [us]"
                       << "|grad|/|fval|"
                       << "#fails"
                       << "#iters"
                       << "#fvals"
                       << "#grads";

        thread_pool_t pool;
        thread_pool_t::mutex_t mutex;

        for (batch_optimizer optimizer : optimizers)
                for (optimize::ls_initializer ls_initializer : ls_initializers)
                        for (optimize::ls_strategy ls_strategy : ls_strategies)
        {
                stats_t<scalar_t> times;
                stats_t<scalar_t> crits;
                stats_t<scalar_t> fails;
                stats_t<scalar_t> iters;
                stats_t<scalar_t> fvals;
                stats_t<scalar_t> grads;

                thread_loopi(trials, pool, [&] (size_t t)
                {
                        const vector_t& x0 = x0s[t];

                        // check gradients
                        const opt_problem_t problem(fn_size, fn_fval, fn_grad);
                        if (problem.grad_accuracy(x0) > math::epsilon2<scalar_t>())
                        {
                                const thread_pool_t::lock_t lock(mutex);

                                log_error() << "invalid gradient for problem [" << problem_name << "]!";
                        }

                        // optimize
                        const ncv::timer_t timer;

                        const opt_state_t state = ncv::minimize(
                                fn_size, fn_fval, fn_grad, nullptr, nullptr, nullptr,
                                x0, optimizer, iterations, epsilon, ls_initializer, ls_strategy);

                        const scalar_t crit = state.convergence_criteria();

                        // update stats
                        const thread_pool_t::lock_t lock(mutex);

                        times(timer.microseconds());
                        crits(crit);
                        iters(state.n_iterations());
                        fvals(state.n_fval_calls());
                        grads(state.n_grad_calls());

                        fails(!state.converged(epsilon) ? 1.0 : 0.0);
                });

                // update per-problem table
                const string_t name =
                        text::to_string(optimizer) + "[" +
                        text::to_string(ls_initializer) + "][" +
                        text::to_string(ls_strategy) + "]";

                table.append(name)
                        << static_cast<int>(fvals.sum() + 2 * grads.sum()) / trials
                        << times.avg()
                        << crits.avg()
                        << static_cast<int>(fails.sum())
                        << iters.avg()
                        << fvals.avg()
                        << grads.avg();

                // update global statistics
                optimizer_stat_t& stat = optimizer_stats[name];
                stat.m_time(times.avg());
                stat.m_crits(crits.avg());
                stat.m_fails(fails.sum());
                stat.m_iters(iters.avg());
                stat.m_fvals(fvals.avg());
                stat.m_grads(grads.avg());
        }

        // print stats
        table.sort_as_number_ascending(3);
        table.print(std::cout);
}

static void check_problems(const std::vector<ncv::function_t>& funcs)
{
        for (const ncv::function_t& func : funcs)
        {
                check_problem(func.m_name, func.m_opsize, func.m_opfval, func.m_opgrad, func.m_solutions);
        }
}

int main(int argc, char *argv[])
{
        using namespace ncv;

        check_problems(ncv::make_beale_funcs());
        check_problems(ncv::make_booth_funcs());
        check_problems(ncv::make_matyas_funcs());
        check_problems(ncv::make_trid_funcs(1024));
        check_problems(ncv::make_sphere_funcs(1024));
        check_problems(ncv::make_powell_funcs(1024));
        check_problems(ncv::make_mccormick_funcs());
        check_problems(ncv::make_himmelblau_funcs());
        check_problems(ncv::make_rosenbrock_funcs(7));
        check_problems(ncv::make_3hump_camel_funcs());
        check_problems(ncv::make_dixon_price_funcs(1024));
        check_problems(ncv::make_sum_squares_funcs(1024));
        check_problems(ncv::make_goldstein_price_funcs());
        check_problems(ncv::make_rotated_ellipsoid_funcs(1024));

        // show global statistics
        tabulator_t table(text::resize("optimizer", 32));
        table.header() << "cost"
                       << "time [us]"
                       << "|grad|/|fval|"
                       << "#fails"
                       << "#iters"
                       << "#fvals"
                       << "#grads";

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
                                   << stat.m_grads.sum();
        }

        table.sort_as_number_ascending(3);
        table.print(std::cout);

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}

