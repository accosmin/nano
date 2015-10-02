#include "minimize.h"
#include "string.h"
#include "min/batch_gd.hpp"
#include "min/batch_cgd.hpp"
#include "min/batch_lbfgs.hpp"
#include "min/stoch_ag.hpp"
#include "min/stoch_sg.hpp"
#include "min/stoch_sga.hpp"
#include "min/stoch_sia.hpp"
#include "min/stoch_adagrad.hpp"
#include "min/stoch_adadelta.hpp"
#include "math/tune_fixed.hpp"

//#define USE_SSE2
//#define LBFGS_FLOAT     64
//#include "lbfgs/lbfgs.h"

namespace ncv
{
//        namespace liblbfgs
//        {
//                struct param_t
//                {
//                        param_t(const opt_problem_t& problem, const vector_t& x0)
//                                :       m_problem(problem),
//                                        m_state(problem, x0)
//                        {
//                        }

//                        const opt_problem_t&    m_problem;
//                        opt_state_t             m_state;
//                };

//                static lbfgsfloatval_t evaluate(
//                        void* instance, const lbfgsfloatval_t* x, lbfgsfloatval_t* g,
//                        const int n, const lbfgsfloatval_t step)
//                {
//                        const param_t* param = reinterpret_cast<param_t*>(instance);

//                        const vector_t vx = tensor::map_vector(x, n);
//                        vector_t gx(n);

//                        const scalar_t fx = (param->m_problem)(vx, gx);

//                        tensor::map_vector(g, n) = gx;

//                        return fx;
//                }

//                static int progress(
//                        void* instance, const lbfgsfloatval_t* x, const lbfgsfloatval_t* g,
//                        const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
//                        const lbfgsfloatval_t step, int n, int k, int ls)
//                {
//                        param_t* param = reinterpret_cast<param_t*>(instance);

//                        param->m_state.m_iterations = k;
//                        param->m_state.m_n_fvals = param->m_problem.n_fval_calls();
//                        param->m_state.m_n_grads = param->m_problem.n_grad_calls();

//                        param->m_state.x = tensor::map_vector(x, n);
//                        param->m_state.g = tensor::map_vector(g, n);
//                        param->m_state.f = fx;

//                        return 0;
//                }

//                opt_state_t minimize(
//                        const opt_problem_t& problem,
//                        const vector_t& x0, size_t iterations, scalar_t epsilon, size_t history_size)
//                {
//                        lbfgs_parameter_t lbfgsparams;
//                        lbfgs_parameter_init(&lbfgsparams);
//                        lbfgsparams.epsilon = epsilon;
//                        lbfgsparams.m = static_cast<int>(history_size);
//                        lbfgsparams.max_iterations = static_cast<int>(iterations);

//                        const int n = static_cast<int>(problem.size());

//                        param_t ret(problem, x0);

//                        lbfgsfloatval_t fx;
//                        lbfgsfloatval_t* x = lbfgs_malloc(n);
//                        tensor::map_vector(x, n) = x0;

//                        lbfgs(n, x, &fx, evaluate, progress, (void*)&ret, &lbfgsparams);

//                        lbfgs_free(x);

//                        return ret.m_state;
//                }
//        }

        opt_state_t minimize(
                const opt_problem_t& problem,
                const opt_opulog_t& fn_ulog,
                const opt_vector_t& x0,
                min::batch_optimizer optimizer, size_t iterations, opt_scalar_t epsilon, size_t history_size)
        {
                switch (optimizer)
                {
                case min::batch_optimizer::LBFGS:
                        return minimize(problem, fn_ulog, x0, optimizer, iterations, epsilon,
                                        min::ls_initializer::unit, min::ls_strategy::interpolation,
                                        history_size);

                case min::batch_optimizer::CGD:
                        return minimize(problem, fn_ulog, x0, min::batch_optimizer::CGD_PRP, iterations, epsilon,
                                        history_size);

                case min::batch_optimizer::CGD_PRP:
                        return minimize(problem, fn_ulog, x0, optimizer, iterations, epsilon,
                                        min::ls_initializer::quadratic, min::ls_strategy::interpolation,
                                        history_size);

                case min::batch_optimizer::CGD_CD:
                        return minimize(problem, fn_ulog, x0, optimizer, iterations, epsilon,
                                        min::ls_initializer::unit, min::ls_strategy::interpolation,
                                        history_size);

                case min::batch_optimizer::CGD_DY:
                        return minimize(problem, fn_ulog, x0, optimizer, iterations, epsilon,
                                        min::ls_initializer::quadratic, min::ls_strategy::backtrack_wolfe,
                                        history_size);

                case min::batch_optimizer::CGD_FR:
                        return minimize(problem, fn_ulog, x0, optimizer, iterations, epsilon,
                                        min::ls_initializer::quadratic, min::ls_strategy::backtrack_armijo,
                                        history_size);

                case min::batch_optimizer::CGD_HS:
                        return minimize(problem, fn_ulog, x0, optimizer, iterations, epsilon,
                                        min::ls_initializer::quadratic, min::ls_strategy::backtrack_wolfe,
                                        history_size);

                case min::batch_optimizer::CGD_LS:
                        return minimize(problem, fn_ulog, x0, optimizer, iterations, epsilon,
                                        min::ls_initializer::quadratic, min::ls_strategy::interpolation,
                                        history_size);

                case min::batch_optimizer::CGD_N:                
                        return minimize(problem, fn_ulog, x0, optimizer, iterations, epsilon,
                                        min::ls_initializer::quadratic, min::ls_strategy::interpolation,
                                        history_size);

                case min::batch_optimizer::CGD_DYCD:
                        return minimize(problem, fn_ulog, x0, optimizer, iterations, epsilon,
                                        min::ls_initializer::unit, min::ls_strategy::interpolation,
                                        history_size);

                case min::batch_optimizer::CGD_DYHS:
                        return minimize(problem, fn_ulog, x0, optimizer, iterations, epsilon,
                                        min::ls_initializer::quadratic, min::ls_strategy::interpolation,
                                        history_size);

                case min::batch_optimizer::GD:
                default:
                        return minimize(problem, fn_ulog, x0, optimizer, iterations, epsilon,
                                        min::ls_initializer::quadratic, min::ls_strategy::backtrack_wolfe,
                                        history_size);
                }
        }

        opt_state_t minimize(
                const opt_problem_t& problem,
                const opt_opulog_t& fn_ulog,
                const opt_vector_t& x0,
                min::batch_optimizer optimizer, size_t iterations, opt_scalar_t epsilon,
                min::ls_initializer lsinit, min::ls_strategy lsstrat,
                size_t history_size)
        {
                switch (optimizer)
                {
//                case min::batch_optimizer::libLBFGS:
//                        return liblbfgs::minimize(problem, x0, iterations, epsilon, history_size);

                case min::batch_optimizer::LBFGS:
                        return  min::batch_lbfgs_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, history_size, fn_ulog)
                                (problem, x0);

                case min::batch_optimizer::CGD:
                        return  min::batch_cgd_prp_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, fn_ulog)
                                (problem, x0);

                case min::batch_optimizer::CGD_CD:
                        return  min::batch_cgd_cd_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, fn_ulog)
                                (problem, x0);

                case min::batch_optimizer::CGD_DY:
                        return  min::batch_cgd_dy_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, fn_ulog)
                                (problem, x0);

                case min::batch_optimizer::CGD_FR:
                        return  min::batch_cgd_fr_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, fn_ulog)
                                (problem, x0);

                case min::batch_optimizer::CGD_HS:
                        return  min::batch_cgd_hs_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, fn_ulog)
                                (problem, x0);

                case min::batch_optimizer::CGD_LS:
                        return  min::batch_cgd_ls_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, fn_ulog)
                                (problem, x0);

                case min::batch_optimizer::CGD_N:
                        return  min::batch_cgd_n_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, fn_ulog)
                                (problem, x0);

                case min::batch_optimizer::CGD_PRP:
                        return  min::batch_cgd_prp_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, fn_ulog)
                                (problem, x0);

                case min::batch_optimizer::CGD_DYCD:
                        return  min::batch_cgd_dycd_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, fn_ulog)
                                (problem, x0);

                case min::batch_optimizer::CGD_DYHS:
                        return  min::batch_cgd_dyhs_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, fn_ulog)
                                (problem, x0);

                case min::batch_optimizer::GD:
                default:
                        return  min::batch_gd_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, fn_ulog)
                                (problem, x0);
                }
        }

        opt_state_t minimize(
                const opt_problem_t& problem,
                const opt_opulog_t& fn_ulog,
                const opt_vector_t& x0,
                min::stoch_optimizer optimizer, size_t epochs, size_t epoch_size, opt_scalar_t alpha0, opt_scalar_t decay)
        {
                switch (optimizer)
                {
                case min::stoch_optimizer::SGA:
                        return  min::stoch_sga_t<opt_problem_t>
                                (epochs, epoch_size, alpha0, decay, fn_ulog)
                                (problem, x0);

                case min::stoch_optimizer::SIA:
                        return  min::stoch_sia_t<opt_problem_t>
                                (epochs, epoch_size, alpha0, decay, fn_ulog)
                                (problem, x0);

                case min::stoch_optimizer::AG:
                        return  min::stoch_ag_t<opt_problem_t>
                                (epochs, epoch_size, alpha0, decay, fn_ulog)
                                (problem, x0);

                case min::stoch_optimizer::AGGR:
                        return  min::stoch_aggr_t<opt_problem_t>
                                (epochs, epoch_size, alpha0, decay, fn_ulog)
                                (problem, x0);

                case min::stoch_optimizer::ADAGRAD:
                        return  min::stoch_adagrad_t<opt_problem_t>
                                (epochs, epoch_size, alpha0, decay, fn_ulog)
                                (problem, x0);

                case min::stoch_optimizer::ADADELTA:
                        return  min::stoch_adadelta_t<opt_problem_t>
                                (epochs, epoch_size, alpha0, decay, fn_ulog)
                                (problem, x0);

                case min::stoch_optimizer::SG:
                default:
                        return  min::stoch_sg_t<opt_problem_t>
                                (epochs, epoch_size, alpha0, decay, fn_ulog)
                                (problem, x0);
                }
        }

        std::vector<opt_scalar_t> tunable_alphas(min::stoch_optimizer optimizer)
        {
                switch (optimizer)
                {
                case min::stoch_optimizer::ADADELTA:
                        return { 0.0 };

                default:
                        return { 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0 };
                }
        }

        std::vector<opt_scalar_t> tunable_decays(min::stoch_optimizer optimizer)
        {
                switch (optimizer)
                {
                case min::stoch_optimizer::AG:
                case min::stoch_optimizer::AGGR:
                case min::stoch_optimizer::ADAGRAD:
                case min::stoch_optimizer::ADADELTA:
                        return { 1.00 };

                default:
                        return { 0.50, 0.75 };
                }
        }

        void tune_stochastic(
                const opt_problem_t& problem, const opt_vector_t& x0,
                min::stoch_optimizer optimizer, opt_size_t epoch_size,
                opt_scalar_t& best_alpha0, opt_scalar_t& best_decay)
        {
                const auto alphas = tunable_alphas(optimizer);
                const auto decays = tunable_decays(optimizer);

                const auto op = [&] (const opt_scalar_t alpha, const opt_scalar_t decay)
                {
                        const auto state = ncv::minimize(problem, nullptr, x0, optimizer, 1, epoch_size, alpha, decay);
                        const auto valid = std::isfinite(state.f);
                        return valid ? state.f : std::numeric_limits<opt_scalar_t>::max();
                };

                const auto config = math::tune_fixed(op, alphas, decays);
                best_alpha0 = std::get<1>(config);
                best_decay = std::get<2>(config);
        }
}
	
