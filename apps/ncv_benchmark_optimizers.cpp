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

#include "libnanocv/functions/function_beale.h"
#include "libnanocv/functions/function_booth.h"
#include "libnanocv/functions/function_sphere.h"
#include "libnanocv/functions/function_matyas.h"
#include "libnanocv/functions/function_ellipse.h"
#include "libnanocv/functions/function_mccormick.h"
#include "libnanocv/functions/function_himmelblau.h"
#include "libnanocv/functions/function_rosenbrock.h"
#include "libnanocv/functions/function_3hump_camel.h"
#include "libnanocv/functions/function_goldstein_price.h"

#include <map>
#include <tuple>

using namespace ncv;

struct optimizer_stat_t
{
        stats_t<scalar_t>       m_time;
        stats_t<scalar_t>       m_fails_2;
        stats_t<scalar_t>       m_fails_3;
        stats_t<scalar_t>       m_fails_4;
        stats_t<scalar_t>       m_fails_5;
        stats_t<scalar_t>       m_fails_6;
        stats_t<scalar_t>       m_iterations;
        stats_t<scalar_t>       m_func_evals;
        stats_t<scalar_t>       m_grad_evals;
};

std::map<string_t, optimizer_stat_t> optimizer_stats;

static void sort_desc(tabulator_t& table, size_t column)
{
        table.sort(column, [] (const string_t& value1, const string_t& value2)
        {
                return text::from_string<scalar_t>(value1) > text::from_string<scalar_t>(value2);
        });
}

static void sort_asc(tabulator_t& table, size_t column)
{
        table.sort(column, [] (const string_t& value1, const string_t& value2)
        {
                return text::from_string<scalar_t>(value1) < text::from_string<scalar_t>(value2);
        });
}

static void check_problem(
        const string_t& problem_name,
        const opt_opsize_t& fn_size, const opt_opfval_t& fn_fval, const opt_opgrad_t& fn_grad,
        const std::vector<std::pair<vector_t, scalar_t>>&)
{
        const size_t iterations = 4 * 1024;
        const scalar_t epsilon = math::epsilon0<scalar_t>();

        const size_t trials = 1024;

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
                batch_optimizer::GD,
                batch_optimizer::CGD_CD,
                batch_optimizer::CGD_DY,
                batch_optimizer::CGD_FR,
                batch_optimizer::CGD_HS,
                batch_optimizer::CGD_LS,
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
                optimize::ls_strategy::backtrack_armijo,
                optimize::ls_strategy::backtrack_wolfe,
                optimize::ls_strategy::backtrack_strong_wolfe,
                optimize::ls_strategy::interpolation_bisection,
                optimize::ls_strategy::interpolation_cubic
        };

        tabulator_t table(text::resize(problem_name, 32));
        table.header() << "speed"
                       << "time [us]"
                       << "|grad|"
                       << "#>e-2"
                       << "#>e-3"
                       << "#>e-4"
                       << "#>e-5"
                       << "#>e-6"
                       << "iters"
                       << "#funcs"
                       << "#grads";

        thread_pool_t pool;
        thread_pool_t::mutex_t mutex;

        for (batch_optimizer optimizer : optimizers)
                for (optimize::ls_initializer ls_initializer : ls_initializers)
                        for (optimize::ls_strategy ls_strategy : ls_strategies)
        {
                stats_t<scalar_t> grads;
                stats_t<scalar_t> times;
                stats_t<scalar_t> fails_2;
                stats_t<scalar_t> fails_3;
                stats_t<scalar_t> fails_4;
                stats_t<scalar_t> fails_5;
                stats_t<scalar_t> fails_6;
                stats_t<scalar_t> opti_iters;
                stats_t<scalar_t> func_evals;
                stats_t<scalar_t> grad_evals;

                thread_loopi(trials, pool, [&] (size_t t)
                {
                        const vector_t& x0 = x0s[t];

                        // optimize
                        const ncv::timer_t timer;

                        const opt_state_t state = ncv::minimize(
                                fn_size, fn_fval, fn_grad, nullptr, nullptr, nullptr,
                                x0, optimizer, iterations, epsilon, ls_initializer, ls_strategy);

                        const scalar_t gnorm = state.g.lpNorm<Eigen::Infinity>();
                        const scalar_t convc = state.convergence_criteria();

                        // update stats
                        const thread_pool_t::lock_t lock(mutex);

                        grads(gnorm);
                        times(timer.microseconds());
                        opti_iters(state.n_iterations());
                        func_evals(state.n_fval_calls());
                        grad_evals(state.n_grad_calls());

                        fails_2(convc > 1e-2 ? 1.0 : 0.0);
                        fails_3(convc > 1e-3 ? 1.0 : 0.0);
                        fails_4(convc > 1e-4 ? 1.0 : 0.0);
                        fails_5(convc > 1e-5 ? 1.0 : 0.0);
                        fails_6(convc > 1e-6 ? 1.0 : 0.0);
                });

                // optimization speed: convergence / #iterations
                const scalar_t speed =
                        -std::log(math::clamp(grads.avg(), epsilon, 1.0 - epsilon)) /
                        (1 + func_evals.avg()/* + 2 * grad_evals.avg()*/);

                // update per-problem table
                const string_t name =
                        text::to_string(optimizer) + "[" +
                        text::to_string(ls_initializer) + "][" +
                        text::to_string(ls_strategy) + "]";

                table.append(name)
                        << speed
                        << times.avg()
                        << grads.avg()
                        << static_cast<int>(fails_2.sum())
                        << static_cast<int>(fails_3.sum())
                        << static_cast<int>(fails_4.sum())
                        << static_cast<int>(fails_5.sum())
                        << static_cast<int>(fails_6.sum())
                        << opti_iters.avg()
                        << func_evals.avg()
                        << grad_evals.avg();

                // update global statistics
                optimizer_stat_t& stat = optimizer_stats[name];
                stat.m_time(times.avg());
                stat.m_fails_2(fails_2.sum());
                stat.m_fails_3(fails_3.sum());
                stat.m_fails_4(fails_4.sum());
                stat.m_fails_5(fails_5.sum());
                stat.m_fails_6(fails_6.sum());
                stat.m_iterations(opti_iters.avg());
                stat.m_func_evals(func_evals.avg());
                stat.m_grad_evals(grad_evals.avg());
        }

        // print stats
        sort_desc(table, 0);
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

//        // Sphere function
//        check_problems(ncv::make_sphere_funcs(16));

//        // Ellipse function
//        check_problems(ncv::make_ellipse_funcs(16));

        // Rosenbrock function
        check_problems(ncv::make_rosenbrock_funcs(7));

//        // Beale function
//        check_problems(ncv::make_beale_funcs());

        // Goldstein-Price function
        check_problems(ncv::make_goldstein_price_funcs());

        // Booth function
        check_problems(ncv::make_booth_funcs());

        // Matyas function
        check_problems(ncv::make_matyas_funcs());

        // Himmelblau function
        check_problems(ncv::make_himmelblau_funcs());

        // 3Hump camel function
        check_problems(ncv::make_3hump_camel_funcs());

        // McCormick function
        check_problems(ncv::make_mccormick_funcs());

        // show global statistics
        tabulator_t table("optimizer");
        table.header() << "time [us]"
                       << ">e-2"
                       << ">e-3"
                       << ">e-4"
                       << ">e-5"
                       << ">e-6"
                       << "iters"
                       << "#funcs"
                       << "#grads";

        for (const auto& it : optimizer_stats)
        {
                const string_t& name = it.first;
                const optimizer_stat_t& stat = it.second;

                table.append(name) << stat.m_time.sum()
                                   << static_cast<int>(stat.m_fails_2.sum())
                                   << static_cast<int>(stat.m_fails_3.sum())
                                   << static_cast<int>(stat.m_fails_4.sum())
                                   << static_cast<int>(stat.m_fails_5.sum())
                                   << static_cast<int>(stat.m_fails_6.sum())
                                   << stat.m_iterations.sum()
                                   << stat.m_func_evals.sum()
                                   << stat.m_grad_evals.sum();
        }

        sort_asc(table, 1);
        table.print(std::cout);

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}

