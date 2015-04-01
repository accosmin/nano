#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_optimizers"

#include <boost/test/unit_test.hpp>
#include "libnanocv/minimize.h"
#include "libnanocv/math/abs.hpp"
#include "libnanocv/util/timer.h"
#include "libnanocv/math/math.hpp"
#include "libnanocv/util/logger.h"
#include "libnanocv/util/stats.hpp"
#include "libnanocv/util/random.hpp"
#include "libnanocv/math/epsilon.hpp"

#include "libnanocv/functions/function_beale.h"
#include "libnanocv/functions/function_booth.h"
#include "libnanocv/functions/function_sphere.h"
#include "libnanocv/functions/function_matyas.h"
#include "libnanocv/functions/function_mccormick.h"
#include "libnanocv/functions/function_himmelblau.h"
#include "libnanocv/functions/function_rosenbrock.h"
#include "libnanocv/functions/function_3hump_camel.h"
#include "libnanocv/functions/function_sum_squares.h"
#include "libnanocv/functions/function_goldstein_price.h"
#include "libnanocv/functions/function_rotated_ellipsoid.h"

namespace test
{
        using namespace ncv;

        static void check_solution(const string_t& problem_name, const string_t& optimizer_name,
                const opt_state_t& state, const std::vector<std::pair<vector_t, scalar_t>>& solutions)
        {
                // Check convergence
                BOOST_CHECK_LE(state.g.lpNorm<Eigen::Infinity>(), math::epsilon3<scalar_t>());

                // Find the closest solution
                size_t best_index = std::string::npos;
                scalar_t best_distance = std::numeric_limits<scalar_t>::max();

                for (size_t index = 0; index < solutions.size(); index ++)
                {
                        const scalar_t distance = (state.x - solutions[index].first).lpNorm<Eigen::Infinity>();
                        if (distance < best_distance)
                        {
                                best_distance = distance;
                                best_index = index;
                        }
                }

                // Check accuracy
                BOOST_CHECK_LT(best_index, solutions.size());
                if (best_index < solutions.size())
                {
                        const scalar_t dfx = math::abs(state.f - solutions[best_index].second);
                        const scalar_t dx = (state.x - solutions[best_index].first).lpNorm<Eigen::Infinity>();

                        BOOST_CHECK_LE(dfx, math::epsilon3<scalar_t>());
                        BOOST_CHECK_LE(dx, math::epsilon3<scalar_t>());

//                        if (dx > math::epsilon3<scalar_t>())
//                        {
//                                log_info() << problem_name
//                                           << ", x = (" << state.x.transpose() << ")"
//                                           << ", dx = " << dx
//                                           << ", x0 = (" << solutions[best_index].first.transpose() << ")"
//                                           << ", fx = " << state.f
//                                           << ", gx = " << state.g.lpNorm<Eigen::Infinity>();
//                        }
                }
        }

        static void check_problem(
                const string_t& problem_name,
                const opt_opsize_t& fn_size, const opt_opfval_t& fn_fval, const opt_opgrad_t& fn_grad,
                const std::vector<std::pair<vector_t, scalar_t>>& solutions)
        {
                const size_t iterations = 64 * 1024;
                const scalar_t epsilon = math::epsilon2<scalar_t>();
                const size_t trials = 8 * 1024;

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
                        batch_optimizer::CGD,
                        batch_optimizer::LBFGS
                };

                for (batch_optimizer optimizer : optimizers)
                {
                        for (size_t t = 0; t < trials; t ++)
                        {
                                const vector_t& x0 = x0s[t];

                                // check gradient
                                const opt_problem_t problem(fn_size, fn_fval, fn_grad);
                                BOOST_CHECK_LE(problem.grad_accuracy(x0), math::epsilon2<scalar_t>());

                                // optimize
                                const opt_state_t state = ncv::minimize(
                                        fn_size, fn_fval, fn_grad, nullptr, nullptr, nullptr,
                                        x0, optimizer, iterations, epsilon);

                                // check solution
                                check_solution(problem_name, text::to_string(optimizer), state, solutions);
                        }
                }
        }

        static void check_problems(const std::vector<test::function_t>& funcs)
        {
                BOOST_CHECK_EQUAL(funcs.empty(), false);

                for (const test::function_t& func : funcs)
                {
                        test::check_problem(func.m_name, func.m_opsize, func.m_opfval, func.m_opgrad, func.m_solutions);
                }
        }
}

BOOST_AUTO_TEST_CASE(test_optimizers)
{
        using namespace ncv;

        // Sphere function
        test::check_problems(test::make_sphere_funcs(16));

        // Sum of squares function
        test::check_problems(test::make_sum_squares_funcs(16));

        // Rotated ellipsoid function
        test::check_problems(test::make_rotated_ellipsoid_funcs(16));

        // Rosenbrock function
        test::check_problems(test::make_rosenbrock_funcs());

        // Booth function
        test::check_problems(test::make_booth_funcs());

        // Matyas function
        test::check_problems(test::make_matyas_funcs());

        // Himmelblau function
        test::check_problems(test::make_himmelblau_funcs());
}

