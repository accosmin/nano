#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_optimizers"

#include <boost/test/unit_test.hpp>
#include "libnanocv/optimize.h"
#include "libnanocv/util/abs.hpp"
#include "libnanocv/util/math.hpp"
#include "libnanocv/util/logger.h"
#include "libnanocv/util/random.hpp"
#include "libnanocv/util/epsilon.hpp"

namespace test
{
        using namespace ncv;

        void check_solution(const string_t& problem_name, const string_t& optimizer_name,
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
                BOOST_CHECK_LE(best_index, solutions.size());
                if (best_index < solutions.size())
                {
                        const scalar_t dfx = math::abs(state.f - solutions[best_index].second);
                        const scalar_t dx = (state.x - solutions[best_index].first).lpNorm<Eigen::Infinity>();

                        BOOST_CHECK_LE(dfx, math::epsilon3<scalar_t>());
                        BOOST_CHECK_LE(dx, math::epsilon3<scalar_t>());

//                        if (dfx > math::epsilon3<scalar_t>())
//                        {
//                                log_error() << "fx = " << state.f << ", x = " << state.x.transpose()
//                                            << "@" << problem_name << "/" << optimizer_name;
//                        }
//                        if (dx > math::epsilon3<scalar_t>())
//                        {
//                                log_error() << "fx = " << state.f << ", x = " << state.x.transpose()
//                                            << "@" << problem_name << "/" << optimizer_name;
//                        }
                }
        }

        void check_problem(
                const string_t& problem_name,
                const opt_opsize_t& fn_size, const opt_opfval_t& fn_fval, const opt_opgrad_t& fn_grad,
                const std::vector<std::pair<vector_t, scalar_t>>& solutions)
        {
                const size_t iterations = 128 * 1024;
                const scalar_t epsilon = std::numeric_limits<scalar_t>::epsilon();
                const size_t history = 6;

                const size_t trials = 16;

                const size_t dims = fn_size();

                const auto optimizers =
                {
                        batch_optimizer::GD,
                        batch_optimizer::CGD,
//                        batch_optimizer::CGD_CD,
                        batch_optimizer::CGD_DY,
//                        batch_optimizer::CGD_FR,
//                        batch_optimizer::CGD_HS,
                        batch_optimizer::CGD_LS,
                        batch_optimizer::CGD_N,
                        batch_optimizer::CGD_PR,
                        batch_optimizer::LBFGS
                };

                for (size_t t = 0; t < trials; t ++)
                {
                        random_t<scalar_t> rgen(-1.0, +1.0);

                        vector_t x0(dims);
                        rgen(x0.data(), x0.data() + x0.size());

                        // check gradient
                        const opt_problem_t fval_problem(fn_size, fn_fval);
                        const opt_problem_t grad_problem(fn_size, fn_fval, fn_grad);

                        vector_t fval_gx0, grad_gx0;
                        fval_problem(x0, fval_gx0);
                        grad_problem(x0, grad_gx0);

                        BOOST_CHECK_LE((fval_gx0 - grad_gx0).lpNorm<Eigen::Infinity>(), math::epsilon3<scalar_t>());

                        // optimize & check solutions
                        for (batch_optimizer optimizer : optimizers)
                        {
                                const opt_state_t state = ncv::minimize(
                                        fn_size, fn_fval, fn_grad, nullptr, nullptr, nullptr,
                                        x0, optimizer, iterations, epsilon, history);

//                                log_info() << "[" << problem_name << ", " << (t + 1) << "/" << trials << ", dims = " << dims << "]"
//                                           << ", optimizer = " << text::to_string(optimizer)
//                                           << ", x = " << state.x.transpose()
//                                           << ", fx = " << state.f
//                                           << ", gx = " << state.g.lpNorm<Eigen::Infinity>()
//                                           << ", evals = " << state.m_n_fvals << "/" << state.m_n_grads;

                                check_solution(problem_name, text::to_string(optimizer), state, solutions);
                        }

//                        log_info();
                }
        }
}

BOOST_AUTO_TEST_CASE(test_optimizers)
{
        using namespace ncv;

        // https://en.wikipedia.org/wiki/Test_functions_for_optimization

        // Sphere function
        for (size_t dims = 1; dims <= 8; dims ++)
        {
                const opt_opsize_t fn_size = [=] ()
                {
                        return dims;
                };

                const opt_opfval_t fn_fval = [=] (const vector_t& x)
                {
                        scalar_t fx = 0;
                        for (size_t i = 0; i < dims; i ++)
                        {
                                fx += x(i) * x(i);
                        }

                        return fx;
                };

                const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        gx.resize(dims);
                        for (size_t i = 0; i < dims; i ++)
                        {
                                gx(i) = 2 * x(i);
                        }

                        return fn_fval(x);
                };

                std::vector<std::pair<vector_t, scalar_t>> solutions;
                {
                        solutions.emplace_back(vector_t::Zero(dims), 0);
                }

                test::check_problem("sphere", fn_size, fn_fval, fn_grad, solutions);
        }

        // Rosenbrock function
        for (size_t dims = 2; dims <= 3; dims ++)
        {
                const opt_opsize_t fn_size = [=] ()
                {
                        return dims;
                };

                const opt_opfval_t fn_fval = [=] (const vector_t& x)
                {
                        scalar_t fx = 0;
                        for (size_t i = 0; i + 1 < dims; i ++)
                        {
                                fx += 100.0 * math::square(x(i + 1) - x(i) * x(i)) + math::square(x(i) - 1);
                        }

                        return fx;
                };

                const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        gx.resize(dims);
                        gx.setZero();
                        for (size_t i = 0; i + 1 < dims; i ++)
                        {
                                gx(i) += 2.0 * (x(i) - 1);
                                gx(i) += 100.0 * 2.0 * (x(i + 1) - x(i) * x(i)) * (- 2.0 * x(i));
                                gx(i + 1) += 100.0 * 2.0 * (x(i + 1) - x(i) * x(i));
                        }

                        return fn_fval(x);
                };

                std::vector<std::pair<vector_t, scalar_t>> solutions;
                {
                        solutions.emplace_back(vector_t::Ones(dims), 0);
                        if (dims >= 4 && dims <= 7)
                        {
                                vector_t x = vector_t::Ones(dims);
                                x(0) = -1;
                                solutions.emplace_back(x, 0);
                        }
                }

                test::check_problem("Rosenbrock", fn_size, fn_fval, fn_grad, solutions);
        }

//        // Beale function
//        {
//                const opt_opsize_t fn_size = [=] ()
//                {
//                        return 2;
//                };

//                const opt_opfval_t fn_fval = [=] (const vector_t& x)
//                {
//                        const scalar_t a = x(0), b = x(1);

//                        const scalar_t z0 = 1.5 - a + a * b;
//                        const scalar_t z1 = 2.25 - a + a * b * b;
//                        const scalar_t z2 = 2.625 - a + a * b * b * b;

//                        return z0 * z0 + z1 * z1 + z2 * z2;
//                };

//                const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
//                {
//                        const scalar_t a = x(0), b = x(1);

//                        const scalar_t z0 = 1.5 - a + a * b;
//                        const scalar_t z1 = 2.25 - a + a * b * b;
//                        const scalar_t z2 = 2.625 - a + a * b * b * b;

//                        gx.resize(2);
//                        gx(0) = 2.0 * (z0 * (-1 + b) + z1 * (-1 + b * b) + z2 * (-1 + b * b * b));
//                        gx(1) = 2.0 * (z0 * (a) + z1 * (2 * a * b) + z2 * (3 * a * b * b));

//                        return fn_fval(x);
//                };

//                std::vector<std::pair<vector_t, scalar_t>> solutions;
//                {
//                        vector_t x(2);
//                        x(0) = 3.0;
//                        x(1) = 0.5;
//                        solutions.emplace_back(x, 0);
//                }

//                fixme: more local minimas?!

//                test::check_problem("Beale", fn_size, fn_fval, fn_grad, solutions);
//        }

//        // Goldstein-Price function
//        {
//                const opt_opsize_t fn_size = [=] ()
//                {
//                        return 2;
//                };

//                const opt_opfval_t fn_fval = [=] (const vector_t& x)
//                {
//                        const scalar_t a = x(0), b = x(1);

//                        const scalar_t z0 = 1.0 + a + b;
//                        const scalar_t z1 = 19 - 14 * a + 3 * a * a - 14 * b + 6 * a * b + 3 * b * b;
//                        const scalar_t z2 = 2 * a - 3 * b;
//                        const scalar_t z3 = 18 - 32 * a + 12 * a * a + 48 * b - 36 * a * b + 27 * b * b;

//                        return (1 + z0 * z0 * z1) * (30 + z2 * z2 * z3);
//                };

//                const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
//                {
//                        const scalar_t a = x(0), b = x(1);

//                        const scalar_t z0 = 1.0 + a + b;
//                        const scalar_t z1 = 19 - 14 * a + 3 * a * a - 14 * b + 6 * a * b + 3 * b * b;
//                        const scalar_t z2 = 2 * a - 3 * b;
//                        const scalar_t z3 = 18 - 32 * a + 12 * a * a + 48 * b - 36 * a * b + 27 * b * b;

//                        const scalar_t u = 1 + z0 * z0 * z1;
//                        const scalar_t v = 30 + z2 * z2 * z3;

//                        const scalar_t z0da = 1;
//                        const scalar_t z0db = 1;

//                        const scalar_t z1da = -14 + 6 * a + 6 * b;
//                        const scalar_t z1db = -14 + 6 * a + 6 * b;

//                        const scalar_t z2da = 2;
//                        const scalar_t z2db = -3;

//                        const scalar_t z3da = -32 + 24 * a - 36 * b;
//                        const scalar_t z3db = 48 - 36 * a + 54 * b;

//                        gx.resize(2);
//                        gx(0) = u * (2 * z2 * z2da * z3 + z2 * z2 * z3da) +
//                                v * (2 * z0 * z0da * z1 + z0 * z0 * z1da);
//                        gx(1) = u * (2 * z2 * z2db * z3 + z2 * z2 * z3db) +
//                                v * (2 * z0 * z0db * z1 + z0 * z0 * z1db);

//                        return fn_fval(x);
//                };

//                std::vector<std::pair<vector_t, scalar_t>> solutions;
//                {
//                        vector_t x(2);
//                        x(0) = +0.0;
//                        x(1) = -1.0;
//                        solutions.emplace_back(x, 3.0);
//                }

//                fixme: more local minimas?!

//                test::check_problem("Goldstein-Price", fn_size, fn_fval, fn_grad, solutions);
//        }

        // Booth function
        {
                const opt_opsize_t fn_size = [=] ()
                {
                        return 2;
                };

                const opt_opfval_t fn_fval = [=] (const vector_t& x)
                {
                        const scalar_t a = x(0), b = x(1);

                        const scalar_t u = a + 2 * b - 7;
                        const scalar_t v = 2 * a + b - 5;

                        return u * u + v * v;
                };

                const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        const scalar_t a = x(0), b = x(1);

                        const scalar_t u = a + 2 * b - 7;
                        const scalar_t v = 2 * a + b - 5;

                        gx.resize(2);
                        gx(0) = 2 * u + 2 * v * 2;
                        gx(1) = 2 * u * 2 + 2 * v;

                        return fn_fval(x);
                };

                std::vector<std::pair<vector_t, scalar_t>> solutions;
                {
                        vector_t x(2);
                        x(0) = 1.0;
                        x(1) = 3.0;
                        solutions.emplace_back(x, 0.0);
                }

                test::check_problem("Booth", fn_size, fn_fval, fn_grad, solutions);
        }

        // Matyas function
        {
                const opt_opsize_t fn_size = [=] ()
                {
                        return 2;
                };

                const opt_opfval_t fn_fval = [=] (const vector_t& x)
                {
                        const scalar_t a = x(0), b = x(1);

                        return 0.26 * (a * a + b * b) - 0.48 * a * b;
                };

                const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        const scalar_t a = x(0), b = x(1);

                        gx.resize(2);
                        gx(0) = 0.26 * 2 * a - 0.48 * b;
                        gx(1) = 0.26 * 2 * b - 0.48 * a;

                        return fn_fval(x);
                };

                std::vector<std::pair<vector_t, scalar_t>> solutions;
                {
                        solutions.emplace_back(vector_t::Zero(2), 0.0);
                }

                test::check_problem("Matyas", fn_size, fn_fval, fn_grad, solutions);
        }

        // Himmelblau function
        {
                const opt_opsize_t fn_size = [=] ()
                {
                        return 2;
                };

                const opt_opfval_t fn_fval = [=] (const vector_t& x)
                {
                        const scalar_t a = x(0), b = x(1);

                        const scalar_t u = a * a + b - 11;
                        const scalar_t v = a + b * b - 7;

                        return u * u + v * v;
                };

                const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        const scalar_t a = x(0), b = x(1);

                        const scalar_t u = a * a + b - 11;
                        const scalar_t v = a + b * b - 7;

                        gx.resize(2);
                        gx(0) = 2 * u * 2 * a + 2 * v;
                        gx(1) = 2 * u + 2 * v * 2 * b;

                        return fn_fval(x);
                };

                std::vector<std::pair<vector_t, scalar_t>> solutions;
                {
                        vector_t x(2);
                        x(0) = -0.270845;
                        x(1) = -0.923039;
                        solutions.emplace_back(x, 181.617);
                }
                {
                        vector_t x(2);
                        x(0) = 3.0;
                        x(1) = 2.0;
                        solutions.emplace_back(x, 0);
                }
                {
                        vector_t x(2);
                        x(0) = -2.805118;
                        x(1) = 3.131312;
                        solutions.emplace_back(x, 0);
                }
                {
                        vector_t x(2);
                        x(0) = -3.779310;
                        x(1) = -3.283186;
                        solutions.emplace_back(x, 0);
                }
                {
                        vector_t x(2);
                        x(0) = 3.584428;
                        x(1) = -1.848126;
                        solutions.emplace_back(x, 0);
                }

                test::check_problem("Himmelblau", fn_size, fn_fval, fn_grad, solutions);
        }
}

