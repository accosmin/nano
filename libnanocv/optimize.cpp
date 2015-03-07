#include "optimize.h"
#include "optimize/batch_gd.hpp"
#include "optimize/batch_cgd.hpp"
#include "optimize/batch_lbfgs.hpp"
#include "optimize/stoch_ag.hpp"
#include "optimize/stoch_sg.hpp"
#include "optimize/stoch_sga.hpp"
#include "optimize/stoch_sia.hpp"
#include "optimize/stoch_adagrad.hpp"
#include "optimize/stoch_adadelta.hpp"
#include "libnanocv/util/logger.h"

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
                batch_optimizer optimizer, size_t iterations, scalar_t epsilon,
                optimize::ls_criterion lscrit, optimize::ls_initializer lsinit,
                size_t history_size)
        {
                const opt_problem_t problem(fn_size, fn_fval, fn_grad);

                switch (optimizer)
                {
                case batch_optimizer::LBFGS:
                        return  optimize::batch_lbfgs_t<opt_problem_t>
                                (iterations, epsilon, lscrit, lsinit, history_size, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case batch_optimizer::CGD:
                        return  optimize::batch_cgd_pr_t<opt_problem_t>
                                (iterations, epsilon, lscrit, lsinit, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case batch_optimizer::CGD_CD:
                        return  optimize::batch_cgd_cd_t<opt_problem_t>
                                (iterations, epsilon, lscrit, lsinit, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case batch_optimizer::CGD_DY:
                        return  optimize::batch_cgd_dy_t<opt_problem_t>
                                (iterations, epsilon, lscrit, lsinit, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case batch_optimizer::CGD_FR:
                        return  optimize::batch_cgd_fr_t<opt_problem_t>
                                (iterations, epsilon, lscrit, lsinit, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case batch_optimizer::CGD_HS:
                        return  optimize::batch_cgd_hs_t<opt_problem_t>
                                (iterations, epsilon, lscrit, lsinit, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case batch_optimizer::CGD_LS:
                        return  optimize::batch_cgd_ls_t<opt_problem_t>
                                (iterations, epsilon, lscrit, lsinit, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case batch_optimizer::CGD_N:
                        return  optimize::batch_cgd_n_t<opt_problem_t>
                                (iterations, epsilon, lscrit, lsinit, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case batch_optimizer::CGD_PR:
                        return  optimize::batch_cgd_pr_t<opt_problem_t>
                                (iterations, epsilon, lscrit, lsinit, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case batch_optimizer::GD:
                default:
                        return  optimize::batch_gd_t<opt_problem_t>
                                (iterations, epsilon, lscrit, lsinit, fn_wlog, fn_elog, fn_ulog)
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
                stochastic_optimizer optimizer, size_t epochs, size_t epoch_size, scalar_t alpha0, scalar_t decay)
        {
                const opt_problem_t problem(fn_size, fn_fval, fn_grad);

                switch (optimizer)
                {
                case stochastic_optimizer::SGA:
                        return  optimize::stoch_sga_t<opt_problem_t>
                                (epochs, epoch_size, alpha0, decay, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case stochastic_optimizer::SIA:
                        return  optimize::stoch_sia_t<opt_problem_t>
                                (epochs, epoch_size, alpha0, decay, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case stochastic_optimizer::AG:
                        return  optimize::stoch_ag_t<opt_problem_t>
                                (epochs, epoch_size, alpha0, decay, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case stochastic_optimizer::ADAGRAD:
                        return  optimize::stoch_adagrad_t<opt_problem_t>
                                (epochs, epoch_size, alpha0, decay, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case stochastic_optimizer::ADADELTA:
                        return  optimize::stoch_adadelta_t<opt_problem_t>
                                (epochs, epoch_size, alpha0, decay, fn_wlog, fn_elog, fn_ulog)
                                (problem, x0);

                case stochastic_optimizer::SG:
                default:
                        return  optimize::stoch_sg_t<opt_problem_t>
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
	
