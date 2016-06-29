#include "params.h"

namespace nano
{
        static ls_initializer make_lsinit(const batch_optimizer optimizer)
        {
                switch (optimizer)
                {
                case batch_optimizer::LBFGS:
                        return ls_initializer::unit;

                case batch_optimizer::CGD: // fall through!
                case batch_optimizer::CGD_PRP:
                        return ls_initializer::quadratic;

                case batch_optimizer::CGD_CD:
                        return ls_initializer::quadratic;

                case batch_optimizer::CGD_DY:
                        return ls_initializer::quadratic;

                case batch_optimizer::CGD_FR:
                        return ls_initializer::quadratic;

                case batch_optimizer::CGD_HS:
                        return ls_initializer::quadratic;

                case batch_optimizer::CGD_LS:
                        return ls_initializer::quadratic;

                case batch_optimizer::CGD_N:
                        return ls_initializer::quadratic;

                case batch_optimizer::CGD_DYCD:
                        return ls_initializer::quadratic;

                case batch_optimizer::CGD_DYHS:
                        return ls_initializer::quadratic;

                case batch_optimizer::GD:
                        return ls_initializer::quadratic;

                default:
                        throw std::runtime_error("make_lsinit: optimization method not handled");
                }
        }

        static ls_strategy make_lsstrat(const batch_optimizer optimizer)
        {
                switch (optimizer)
                {
                case batch_optimizer::LBFGS:
                        return ls_strategy::interpolation;

                case batch_optimizer::CGD: // fall through!
                case batch_optimizer::CGD_PRP:
                        return ls_strategy::interpolation;

                case batch_optimizer::CGD_CD:
                        return ls_strategy::interpolation;

                case batch_optimizer::CGD_DY:
                        return ls_strategy::backtrack_wolfe;

                case batch_optimizer::CGD_FR:
                        return ls_strategy::interpolation;

                case batch_optimizer::CGD_HS:
                        return ls_strategy::interpolation;

                case batch_optimizer::CGD_LS:
                        return ls_strategy::interpolation;

                case batch_optimizer::CGD_N:
                        return ls_strategy::interpolation;

                case batch_optimizer::CGD_DYCD:
                        return ls_strategy::backtrack_wolfe;

                case batch_optimizer::CGD_DYHS:
                        return ls_strategy::backtrack_wolfe;

                case batch_optimizer::GD:
                        return ls_strategy::backtrack_wolfe;

                default:
                        throw std::runtime_error("make_lsstrat: optimization method not handled");
                }
        }

        batch_params_t::batch_params_t(
                const std::size_t max_iterations,
                const scalar_t epsilon,
                const batch_optimizer optimizer,
                const ls_initializer lsinit,
                const ls_strategy lsstrat,
                const opulog_t& ulog,
                const std::size_t lbfgs_hsize,
                const scalar_t cgd_orthotest) :
                m_ulog(ulog),
                m_max_iterations(max_iterations),
                m_epsilon(epsilon),
                m_optimizer(optimizer),
                m_ls_initializer(lsinit),
                m_ls_strategy(lsstrat),
                m_lbfgs_hsize(lbfgs_hsize),
                m_cgd_orthotest(cgd_orthotest)
        {
        }

        batch_params_t::batch_params_t(
                const std::size_t max_iterations,
                const scalar_t epsilon,
                const batch_optimizer optimizer,
                const opulog_t& ulog,
                const std::size_t lbfgs_hsize,
                const scalar_t cgd_orthotest) :
                m_ulog(ulog),
                m_max_iterations(max_iterations),
                m_epsilon(epsilon),
                m_optimizer(optimizer),
                m_ls_initializer(make_lsinit(optimizer)),
                m_ls_strategy(make_lsstrat(optimizer)),
                m_lbfgs_hsize(lbfgs_hsize),
                m_cgd_orthotest(cgd_orthotest)
        {
        }
}

