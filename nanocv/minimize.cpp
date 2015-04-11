#include "minimize.h"
#include "optim/batch_gd.hpp"
#include "optim/batch_cgd.hpp"
#include "optim/batch_lbfgs.hpp"
#include "optim/stoch_ag.hpp"
#include "optim/stoch_sg.hpp"
#include "optim/stoch_sga.hpp"
#include "optim/stoch_sia.hpp"
#include "optim/stoch_adagrad.hpp"
#include "optim/stoch_adadelta.hpp"
#include "logger.h"
#include "string.h"

namespace ncv
{
        opt_state_t minimize(
                const opt_opsize_t& fn_size,
                const opt_opfval_t& fn_fval,
                const opt_opgrad_t& fn_grad,
                const opt_opwlog_t& fn_wlog,
                const opt_opelog_t& fn_elog,
                const opt_opulog_t& fn_ulog,
                const vector_t& x0,
                optim::batch_optimizer optimizer, size_t iterations, scalar_t epsilon)
        {
                switch (optimizer)
                {
                case optim::batch_optimizer::LBFGS:
                        return minimize(fn_size, fn_fval, fn_grad, fn_wlog, fn_elog, fn_ulog, x0, optimizer, iterations, epsilon,
                                        optim::ls_initializer::unit, optim::ls_strategy::interpolation);

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
                        return minimize(fn_size, fn_fval, fn_grad, fn_wlog, fn_elog, fn_ulog, x0, optimizer, iterations, epsilon,
                                        optim::ls_initializer::quadratic, optim::ls_strategy::interpolation);

                case optim::batch_optimizer::GD:
                default:
                        return minimize(fn_size, fn_fval, fn_grad, fn_wlog, fn_elog, fn_ulog, x0, optimizer, iterations, epsilon,
                                        optim::ls_initializer::quadratic, optim::ls_strategy::backtrack_wolfe);
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
	
