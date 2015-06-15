#include "minimize.h"
#include "optim/batch/gd.hpp"
#include "optim/batch/cgd.hpp"
#include "optim/batch/lbfgs.hpp"
#include "optim/stochastic/ag.hpp"
#include "optim/stochastic/sg.hpp"
#include "optim/stochastic/sga.hpp"
#include "optim/stochastic/sia.hpp"
#include "optim/stochastic/adagrad.hpp"
#include "optim/stochastic/adadelta.hpp"
#include "logger.h"
#include "string.h"

//#define USE_SSE2
//#define LBFGS_FLOAT     64
//#include "liblbfgs/lbfgs.h"

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
                const opt_opsize_t& fn_size,
                const opt_opfval_t& fn_fval,
                const opt_opgrad_t& fn_grad,
                const opt_opwlog_t& fn_wlog,
                const opt_opelog_t& fn_elog,
                const opt_opulog_t& fn_ulog,
                const vector_t& x0,
                optim::batch_optimizer optimizer, size_t iterations, scalar_t epsilon, size_t history_size)
        {
                switch (optimizer)
                {
                case optim::batch_optimizer::LBFGS:
                        return minimize(fn_size, fn_fval, fn_grad, fn_wlog, fn_elog, fn_ulog, x0,
                                        optimizer, iterations, epsilon,
                                        optim::ls_initializer::unit, optim::ls_strategy::interpolation,
                                        history_size);

                case optim::batch_optimizer::CGD:
                case optim::batch_optimizer::CGD_CD:
                case optim::batch_optimizer::CGD_DY:
                case optim::batch_optimizer::CGD_FR:
                case optim::batch_optimizer::CGD_HS:
                case optim::batch_optimizer::CGD_LS:
                case optim::batch_optimizer::CGD_N:
                case optim::batch_optimizer::CGD_PRP:
                case optim::batch_optimizer::CGD_DYCD:
                case optim::batch_optimizer::CGD_DYHS:
                        return minimize(fn_size, fn_fval, fn_grad, fn_wlog, fn_elog, fn_ulog, x0,
                                        optimizer, iterations, epsilon,
                                        optim::ls_initializer::quadratic, optim::ls_strategy::interpolation,
                                        history_size);

                case optim::batch_optimizer::GD:
                default:
                        return minimize(fn_size, fn_fval, fn_grad, fn_wlog, fn_elog, fn_ulog, x0,
                                        optimizer, iterations, epsilon,
                                        optim::ls_initializer::quadratic, optim::ls_strategy::backtrack_wolfe,
                                        history_size);
                }
        }

        opt_state_t minimize(
                const opt_opsize_t& fn_size,
                const opt_opfval_t& fn_fval,
                const opt_opgrad_t& fn_grad,
                const opt_opwlog_t& fn_wlog,
                const opt_opelog_t& fn_elog,
                const opt_opulog_t& fn_ulog,
                const vector_t& x0,
                optim::batch_optimizer optimizer, size_t iterations, scalar_t epsilon,
                optim::ls_initializer lsinit, optim::ls_strategy lsstrat,
                size_t history_size)
        {
                const opt_problem_t problem(fn_size, fn_fval, fn_grad);

                switch (optimizer)
                {
//                case optim::batch_optimizer::libLBFGS:
//                        return liblbfgs::minimize(problem, x0, iterations, epsilon, history_size);

                case optim::batch_optimizer::LBFGS:
                        return  optim::batch_lbfgs_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, history_size, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case optim::batch_optimizer::CGD:
                        return  optim::batch_cgd_n_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case optim::batch_optimizer::CGD_CD:
                        return  optim::batch_cgd_cd_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case optim::batch_optimizer::CGD_DY:
                        return  optim::batch_cgd_dy_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case optim::batch_optimizer::CGD_FR:
                        return  optim::batch_cgd_fr_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case optim::batch_optimizer::CGD_HS:
                        return  optim::batch_cgd_hs_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case optim::batch_optimizer::CGD_LS:
                        return  optim::batch_cgd_ls_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case optim::batch_optimizer::CGD_N:
                        return  optim::batch_cgd_n_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case optim::batch_optimizer::CGD_PRP:
                        return  optim::batch_cgd_prp_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case optim::batch_optimizer::CGD_DYCD:
                        return  optim::batch_cgd_dycd_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case optim::batch_optimizer::CGD_DYHS:
                        return  optim::batch_cgd_dyhs_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case optim::batch_optimizer::GD:
                default:
                        return  optim::batch_gd_t<opt_problem_t>
                                (iterations, epsilon, lsinit, lsstrat, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);
                }
        }

        opt_state_t minimize(
                const opt_opsize_t& fn_size,
                const opt_opfval_t& fn_fval,
                const opt_opgrad_t& fn_grad,
                const opt_opwlog_t& fn_wlog,
                const opt_opelog_t& fn_elog,
                const opt_opulog_t& fn_ulog,
                const vector_t& x0,
                optim::stoch_optimizer optimizer, size_t epochs, size_t epoch_size, scalar_t alpha0, scalar_t decay)
        {
                const opt_problem_t problem(fn_size, fn_fval, fn_grad);

                switch (optimizer)
                {
                case optim::stoch_optimizer::SGA:
                        return  optim::stoch_sga_t<opt_problem_t>
                                (epochs, epoch_size, alpha0, decay, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case optim::stoch_optimizer::SIA:
                        return  optim::stoch_sia_t<opt_problem_t>
                                (epochs, epoch_size, alpha0, decay, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case optim::stoch_optimizer::AG:
                        return  optim::stoch_ag_t<opt_problem_t>
                                (epochs, epoch_size, alpha0, decay, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case optim::stoch_optimizer::AGGR:
                        return  optim::stoch_aggr_t<opt_problem_t>
                                (epochs, epoch_size, alpha0, decay, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case optim::stoch_optimizer::ADAGRAD:
                        return  optim::stoch_adagrad_t<opt_problem_t>
                                (epochs, epoch_size, alpha0, decay, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case optim::stoch_optimizer::ADADELTA:
                        return  optim::stoch_adadelta_t<opt_problem_t>
                                (epochs, epoch_size, alpha0, decay, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case optim::stoch_optimizer::SG:
                default:
                        return  optim::stoch_sg_t<opt_problem_t>
                                (epochs, epoch_size, alpha0, decay, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);
                }
        }

        opt_opwlog_t make_opwlog()
        {
                return [] (const string_t& message)
                {
                        log_warning() << message;
                };
        }

        opt_opelog_t make_opelog()
        {
                return [] (const string_t& message)
                {
                        log_error() << message;
                };
        }
}
	
