#include "libnanocv/optimize.h"
#include "libnanocv/util/abs.hpp"
#include "libnanocv/util/timer.h"
#include "libnanocv/util/math.hpp"
#include "libnanocv/util/logger.h"
#include "libnanocv/util/stats.hpp"
#include "libnanocv/util/random.hpp"
#include "libnanocv/util/epsilon.hpp"
#include "libnanocv/util/tabulator.h"
#include "libnanocv/util/thread_loop.hpp"

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

using namespace ncv;

static void check_problem(
        const string_t& problem_name,
        const opt_opsize_t& fn_size, const opt_opfval_t& fn_fval, const opt_opgrad_t& fn_grad,
        const std::vector<std::pair<vector_t, scalar_t>>& solutions)
{
        const size_t iterations = 4 * 1024;
        const scalar_t epsilon = std::numeric_limits<scalar_t>::epsilon();

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
                batch_optimizer::CGD_PR,
                batch_optimizer::CGD_N,
                batch_optimizer::LBFGS
        };

        const auto ls_initializers =
        {
                optimize::ls_initializer::unit,
                optimize::ls_initializer::quadratic,
                optimize::ls_initializer::consistent
        };

        tabulator_t table(text::resize(problem_name, 32));
        table.header() << "grad"
                       << "time [us]"
                       << "iterations"
                       << "func evals"
                       << "grad evals";

        thread_pool_t pool;
        thread_pool_t::mutex_t mutex;

        for (batch_optimizer optimizer : optimizers)
                for (optimize::ls_initializer ls_initializer : ls_initializers)
        {
                stats_t<scalar_t> grads;
                stats_t<scalar_t> times;
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
                                x0, optimizer, iterations, epsilon,
                                optimize::ls_criterion::strong_wolfe,
                                ls_initializer);

                        // update stats
                        const thread_pool_t::lock_t lock(mutex);

                        grads(state.g.lpNorm<Eigen::Infinity>());
                        times(timer.microseconds());
                        opti_iters(state.n_iterations());
                        func_evals(state.n_fval_calls());
                        grad_evals(state.n_grad_calls());
                });

                table.append(text::to_string(optimizer) + ":" + text::to_string(ls_initializer))
                        << grads.avg()
                        << times.avg()
                        << opti_iters.avg()
                        << func_evals.avg()
                        << grad_evals.avg();
        }

        // print stats
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

        // Sphere function
        check_problems(ncv::make_sphere_funcs(16));

        // Ellipse function
        check_problems(ncv::make_ellipse_funcs(16));

        // Rosenbrock function
        check_problems(ncv::make_rosenbrock_funcs());

        // Beale function
        check_problems(ncv::make_beale_funcs());

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

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}

