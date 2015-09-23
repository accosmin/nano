#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_optimizers"

#include <boost/test/unit_test.hpp>
#include "math/abs.hpp"
#include "core/logger.h"
#include "core/minimize.h"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "math/epsilon.hpp"
#include "text/to_string.hpp"

#include "func/function_trid.h"
#include "func/function_beale.h"
#include "func/function_booth.h"
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

namespace test
{
        using namespace ncv;

//        static void check_solution(const string_t& problem, const string_t& optimizer,
//                const vector_t& x0, const opt_state_t& state, const std::vector<std::pair<vector_t, scalar_t>>& solutions)
//        {
//                // find the closest solution
//                size_t best_index = std::string::npos;
//                scalar_t best_distance = std::numeric_limits<scalar_t>::max();

//                for (size_t index = 0; index < solutions.size(); index ++)
//                {
//                        const scalar_t distance = (state.x - solutions[index].first).lpNorm<Eigen::Infinity>();
//                        if (distance < best_distance)
//                        {
//                                best_distance = distance;
//                                best_index = index;
//                        }
//                }

//                // check accuracy
//                BOOST_REQUIRE_LT(best_index, solutions.size());

//                const scalar_t df = math::abs(state.f - solutions[best_index].second);
//                const scalar_t dx = (state.x - solutions[best_index].first).lpNorm<Eigen::Infinity>();

//                BOOST_CHECK_LE(df, math::epsilon3<scalar_t>());
//                BOOST_CHECK_LE(dx, math::epsilon3<scalar_t>());

//                // debugging
//                BOOST_CHECK_MESSAGE(
//                        dx < math::epsilon3<scalar_t>(),
//                        "failed (x) after " << state.n_iterations() <<
//                        " iterations for <" << problem << ">, <" << optimizer << "> and <" << x0.transpose() << ">!");
//                BOOST_CHECK_MESSAGE(
//                        df < math::epsilon3<scalar_t>(),
//                        "failed (f) after " << state.n_iterations() <<
//                        " iterations for <" << problem << ">, <" << optimizer << "> and <" << x0.transpose() << ">!");
//        }

        static void check_problem(const test::function_t& func)
        {
                const auto func_name = func.m_name;
                const auto fn_size = func.m_opsize;
                const auto fn_fval = func.m_opfval;
                const auto fn_grad = func.m_opgrad;

                const auto iterations = size_t(1024);
                const auto epsilon = scalar_t(1e-6);
                const auto trials = size_t(1024);

                const auto dims = fn_size();

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

                        min::batch_optimizer::CGD,
                        min::batch_optimizer::CGD_CD,
                        min::batch_optimizer::CGD_DY,
                        min::batch_optimizer::CGD_FR,
                        min::batch_optimizer::CGD_HS,
                        min::batch_optimizer::CGD_LS,
                        min::batch_optimizer::CGD_N,
                        min::batch_optimizer::CGD_PRP,
                        min::batch_optimizer::CGD_DYCD,
                        min::batch_optimizer::CGD_DYHS,

                        min::batch_optimizer::LBFGS
                };

                for (min::batch_optimizer optimizer : optimizers)
                {
                        const auto opt_name = text::to_string(optimizer);

                        for (size_t t = 0; t < trials; t ++)
                        {
                                const auto& x0 = x0s[t];
                                const auto x0t = x0.transpose();
                                const auto f0 = fn_fval(x0);

                                // optimize
                                const auto state = ncv::minimize(
                                        fn_size, fn_fval, fn_grad, nullptr, nullptr, nullptr,
                                        x0, optimizer, iterations, epsilon);

                                const auto it = state.n_iterations();

                                #define NANOCV_TEST_OPTIMIZERS_DESCRIPTION \
                                        ") for (" << func_name << ", " << opt_name << \
                                        ", x0 = [" << x0t << "], " << it << " iterations)"

                                // check function value decrease
                                const auto fx = state.f;
                                const auto fx_thres = f0 - epsilon * math::abs(f0);

                                BOOST_CHECK_MESSAGE(fx < f0,
                                        "decrease failed (" << fx << " < " << f0 <<
                                        NANOCV_TEST_OPTIMIZERS_DESCRIPTION);
                                BOOST_CHECK_MESSAGE(fx < fx_thres,
                                        "sufficient decrease failed (" << fx << " < " << fx_thres <<
                                        NANOCV_TEST_OPTIMIZERS_DESCRIPTION);

                                // check convergence
                                const auto gx = state.convergence_criteria();
                                const auto gx_thres = epsilon;

                                BOOST_CHECK_MESSAGE(gx < gx_thres,
                                        "convergence failed (" << gx << " < " << gx_thres <<
                                        NANOCV_TEST_OPTIMIZERS_DESCRIPTION);

                                // todo: check local minimas (if any known)
                        }
                }
        }

        static void check_problems(const std::vector<test::function_t>& funcs)
        {
                for (const auto& func : funcs)
                {
                        test::check_problem(func);
                }
        }
}

BOOST_AUTO_TEST_CASE(test_optimizers)
{
        using namespace ncv;        

        test::check_problems(ncv::make_beale_funcs());
        test::check_problems(ncv::make_booth_funcs());
        test::check_problems(ncv::make_matyas_funcs());
        test::check_problems(ncv::make_trid_funcs(8));
        test::check_problems(ncv::make_sphere_funcs(8));
        test::check_problems(ncv::make_powell_funcs(8));
        test::check_problems(ncv::make_mccormick_funcs());
        test::check_problems(ncv::make_himmelblau_funcs());
        test::check_problems(ncv::make_rosenbrock_funcs(7));
        test::check_problems(ncv::make_3hump_camel_funcs());
        test::check_problems(ncv::make_dixon_price_funcs(8));
        test::check_problems(ncv::make_sum_squares_funcs(8));
        test::check_problems(ncv::make_goldstein_price_funcs());
        test::check_problems(ncv::make_rotated_ellipsoid_funcs(8));
}

