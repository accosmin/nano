#ifndef NANOCV_OPTIMIZE_OPTIMIZER_CGD_HPP
#define NANOCV_OPTIMIZE_OPTIMIZER_CGD_HPP

#include "ls_wolfe.hpp"
#include <cassert>

namespace ncv
{
        namespace optimize
        {
                namespace detail
                {
                        // these variantions have been implemented following
                        //      "A survey of nonlinear conjugate gradient methods"
                        //      by William W. Hager and Hongchao Zhang

                        ///
                        /// \brief CGD update parameters (Hestenes and Stiefel, 1952)
                        ///
                        template
                        <
                                typename tstate,

                                typename tscalar = typename tstate::tscalar
                        >
                        tscalar step_HS(const tstate& pstate, const tstate& cstate)
                        {
                                const auto& dk = pstate.d;
                                const auto& gk = pstate.g;
                                const auto& gk1 = cstate.g;
                                const auto yk = gk1 - gk;

                                return gk1.dot(yk) / dk.dot(yk);
                        }

                        ///
                        /// \brief CGD update parameters (Fletcher and Reeves, 1964)
                        ///
                        template
                        <
                                typename tstate,

                                typename tscalar = typename tstate::tscalar
                        >
                        tscalar step_FR(const tstate& pstate, const tstate& cstate)
                        {                                
                                const auto& gk = pstate.g;
                                const auto& gk1 = cstate.g;

                                return gk1.squaredNorm() / gk.squaredNorm();
                        }

                        ///
                        /// \brief CGD update parameters (Polak and Ribiere, 1969)
                        ///
                        template
                        <
                                typename tstate,

                                typename tscalar = typename tstate::tscalar
                        >
                        tscalar step_PR(const tstate& pstate, const tstate& cstate)
                        {
                                const auto& gk = pstate.g;
                                const auto& gk1 = cstate.g;
                                const auto yk = gk1 - gk;

                                return gk1.dot(yk) / gk.squaredNorm();
                        }

                        ///
                        /// \brief CGD update parameters (Fletcher - Conjugate Descent, 1987)
                        ///
                        template
                        <
                                typename tstate,

                                typename tscalar = typename tstate::tscalar
                        >
                        tscalar step_CD(const tstate& pstate, const tstate& cstate)
                        {
                                const auto& dk = pstate.d;
                                const auto& gk = pstate.g;
                                const auto& gk1 = cstate.g;

                                return - gk1.squaredNorm() / dk.dot(gk);
                        }

                        ///
                        /// \brief CGD update parameters (Liu and Storey, 1991)
                        ///
                        template
                        <
                                typename tstate,

                                typename tscalar = typename tstate::tscalar
                        >
                        tscalar step_LS(const tstate& pstate, const tstate& cstate)
                        {
                                const auto& dk = pstate.d;
                                const auto& gk = pstate.g;
                                const auto& gk1 = cstate.g;
                                const auto yk = gk1 - gk;

                                return - gk1.dot(yk) / dk.dot(gk);
                        }

                        ///
                        /// \brief CGD update parameters (Dai and Yuan, 1999)
                        ///
                        template
                        <
                                typename tstate,

                                typename tscalar = typename tstate::tscalar
                        >
                        tscalar step_DY(const tstate& pstate, const tstate& cstate)
                        {
                                const auto& dk = pstate.d;
                                const auto& gk = pstate.g;
                                const auto& gk1 = cstate.g;
                                const auto yk = gk1 - gk;

                                return - gk1.squaredNorm() / dk.dot(yk);
                        }

                        ///
                        /// \brief CGD update parameters (Hager and Zhang, 2005)
                        ///
                        template
                        <
                                typename tstate,

                                typename tscalar = typename tstate::tscalar
                        >
                        tscalar step_N(const tstate& pstate, const tstate& cstate)
                        {
                                const auto& dk = pstate.d;
                                const auto& gk = pstate.g;
                                const auto& gk1 = cstate.g;
                                const auto yk = gk1 - gk;
                                const tscalar div = 1 / dk.dot(yk);

                                return (yk - 2 * dk * yk.squaredNorm() * div).dot(gk1 * div);
                        }

                        ///
                        /// \brief conjugate gradient descent starting from the initial value (guess) x0
                        ///
                        template
                        <
                                typename tproblem,
                                typename tcgd_update,

                                // dependent types
                                typename tscalar = typename tproblem::tscalar,
                                typename tsize = typename tproblem::tsize,
                                typename tvector = typename tproblem::tvector,
                                typename tstate = typename tproblem::tstate,

                                typename twlog = typename tproblem::twlog,
                                typename telog = typename tproblem::telog,
                                typename tulog = typename tproblem::tulog
                        >
                        tstate cgd(
                                const tproblem& problem,
                                const tcgd_update& op_update,
                                const tvector& x0,
                                tsize max_iterations,           ///< maximum number of iterations
                                tscalar epsilon,                ///< convergence precision
                                const twlog& op_wlog = twlog(),
                                const telog& op_elog = telog(),
                                const tulog& op_ulog = tulog())
                        {
                                assert(problem.size() == static_cast<tsize>(x0.size()));

                                tstate cstate(problem, x0), pstate = cstate;

                                tscalar ft;
                                tvector gt;

                                // iterate until convergence
                                for (tsize i = 0; i < max_iterations; i ++)
                                {
                                        if (op_ulog)
                                        {
                                                op_ulog(cstate);
                                        }

                                        // check convergence
                                        if (cstate.converged(epsilon))
                                        {
                                                break;
                                        }

                                        // descent direction
                                        if (i == 0)
                                        {
                                                cstate.d = -cstate.g;
                                        }
                                        else
                                        {
                                                const tscalar beta = op_update(pstate, cstate);
                                                cstate.d = -cstate.g + std::max(static_cast<tscalar>(0), beta) * pstate.d;
                                        }

                                        // update solution
                                        const tscalar t = ls_strong_wolfe(problem, cstate, op_wlog, ft, gt, 1e-4, 0.1);
                                        if (t < std::numeric_limits<tscalar>::epsilon())
                                        {
                                                if (op_elog)
                                                {
                                                        op_elog("line-search failed for CGD!");
                                                }
                                                break;
                                        }
                                        pstate = cstate;
                                        cstate.update(problem, t, ft, gt);
                                }

                                return cstate;
                        }
                }

                #define NCV_MAKE_CGD_OPTIMIZER(NAME, STEP) \
                template \
                < \
                        typename tproblem, \
                        \
                        typename tscalar = typename tproblem::tscalar, \
                        typename tsize = typename tproblem::tsize, \
                        typename tvector = typename tproblem::tvector, \
                        typename tstate = typename tproblem::tstate, \
                        \
                        typename twlog = typename tproblem::twlog, \
                        typename telog = typename tproblem::telog, \
                        typename tulog = typename tproblem::tulog \
                > \
                tstate cgd_##NAME( \
                        const tproblem& problem, \
                        const tvector& x0, \
                        tsize max_iterations, \
                        tscalar epsilon, \
                        const twlog& op_wlog = twlog(), \
                        const telog& op_elog = telog(), \
                        const tulog& op_ulog = tulog()) \
                { \
                        return  detail::cgd(problem, detail::step_##STEP<tstate>, \
                                x0, max_iterations, epsilon, op_wlog, op_elog, op_ulog); \
                }

                // instantiate various CGD algorithms
                NCV_MAKE_CGD_OPTIMIZER(hs, HS)
                NCV_MAKE_CGD_OPTIMIZER(fr, FR)
                NCV_MAKE_CGD_OPTIMIZER(pr, PR)
                NCV_MAKE_CGD_OPTIMIZER(cd, CD)
                NCV_MAKE_CGD_OPTIMIZER(ls, LS)
                NCV_MAKE_CGD_OPTIMIZER(dy, DY)
                NCV_MAKE_CGD_OPTIMIZER(n,  N)
        }
}

#endif // NANOCV_OPTIMIZE_OPTIMIZER_CGD_HPP
