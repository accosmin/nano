#ifndef NANOCV_OPTIMIZE_ALGO_HPP
#define NANOCV_OPTIMIZE_ALGO_HPP

#include "core/optimize/result.hpp"
#include <type_traits>
#include <deque>
#include "core/logger.h"        // FIXME: remove this dependency!

namespace ncv
{
        namespace optimize
        {
                template
                <
                        typename tscalar,
                        typename tsize,
                        typename tvalid_tscalar = typename std::enable_if<std::is_floating_point<tscalar>::value>::type,
                        typename tvalid_tsize = typename std::enable_if<std::is_integral<tsize>::value>::type
                >
                class optimizer
                {
                public:

                        typedef problem_t<tscalar, tsize>               tproblem;
                        typedef typename tproblem::vector_t             tvector;
                        typedef state_t<tscalar, tsize>                 tstate;
                        typedef result_t<tscalar, tsize>                tresult;

                        // update logging operator
                        typedef std::function<void(const tresult&)>     op_updated_t;

                        /////////////////////////////////////////////////////////////////////////////////////////////
                        // gradient descent starting from the initial value (guess) x0.
                        /////////////////////////////////////////////////////////////////////////////////////////////

                        static tresult gd(
                                const tproblem& problem,
                                const tvector& x0,
                                tsize max_iterations,          // maximum number of iterations
                                tscalar epsilon,               // convergence precision
                                const op_updated_t& op_updated = op_updated_t())
                        {
                                return _gd(problem, x0, max_iterations, epsilon, op_updated);
                        }

                        /////////////////////////////////////////////////////////////////////////////////////////////
                        // conjugate gradient descent starting from the initial value (guess) x0.
                        /////////////////////////////////////////////////////////////////////////////////////////////

                        static tresult cgd(
                                const tproblem& problem,
                                const tvector& x0,
                                tsize max_iterations,          // maximum number of iterations
                                tscalar epsilon,               // convergence precision
                                const op_updated_t& op_updated = op_updated_t())
                        {
                                return _cgd(problem, x0, max_iterations, epsilon, op_updated);
                        }

                        /////////////////////////////////////////////////////////////////////////////////////////////
                        // limited memory bfgs (l-bfgs) starting from the initial value (guess) x0.
                        /////////////////////////////////////////////////////////////////////////////////////////////

                        static tresult lbfgs(
                                const tproblem& problem,
                                const tvector& x0,
                                tsize max_iterations,          // maximum number of iterations
                                tscalar epsilon,               // convergence precision
                                tsize history_size = 8,        // hessian approximation history size
                                const op_updated_t& op_updated = op_updated_t())
                        {
                                return _lbfgs(problem, x0, max_iterations, epsilon, history_size, op_updated);
                        }

                private:

                        /////////////////////////////////////////////////////////////////////////////////////////////
                        // gradient descent starting from the initial value (guess) x0.
                        /////////////////////////////////////////////////////////////////////////////////////////////

                        static tresult _gd(
                                const tproblem& problem,
                                const tvector& x0,
                                tsize max_iterations,          // maximum number of iterations
                                tscalar epsilon,               // convergence precision
                                const op_updated_t& op_updated = op_updated_t())
                        {
                                assert(problem.size() == math::cast<tsize>(x0.size()));

                                tresult result(problem.size());
                                tstate cstate(problem, x0);

                                // iterate until convergence
                                for (tsize i = 0; i < max_iterations; i ++)
                                {
                                        result.update(problem, cstate);
                                        if (op_updated)
                                        {
                                                op_updated(result);
                                        }

                                        // check convergence
                                        if (cstate.converged(epsilon))
                                        {
                                                break;
                                        }

                                        // descent direction
                                        cstate.d = -cstate.g;

                                        // update solution
                                        const tscalar t = ls_armijo(problem, cstate);
                                        if (t < std::numeric_limits<tscalar>::epsilon())
                                        {
                                                log_warning() << "optimize: line-search failed for GD!";
                                                break;
                                        }
                                        cstate.update(problem, t);
                                }

                                return result;
                        }

                        /////////////////////////////////////////////////////////////////////////////////////////////
                        // conjugate gradient descent starting from the initial value (guess) x0.
                        /////////////////////////////////////////////////////////////////////////////////////////////

                        static tresult _cgd(
                                const tproblem& problem,
                                const tvector& x0,
                                tsize max_iterations,          // maximum number of iterations
                                tscalar epsilon,               // convergence precision
                                const op_updated_t& op_updated = op_updated_t())
                        {
                                assert(problem.size() == math::cast<tsize>(x0.size()));

                                tresult result(problem.size());
                                tstate cstate(problem, x0), pstate = cstate;

                                tscalar ft;
                                tvector gt;

                                // iterate until convergence
                                for (tsize i = 0; i < max_iterations; i ++)
                                {
                                        result.update(problem, cstate);
                                        if (op_updated)
                                        {
                                                op_updated(result);
                                        }

                                        // check convergence
                                        if (cstate.converged(epsilon))
                                        {
                                                break;
                                        }

                                        // descent direction (Polak–Ribière updates)
                                        if (i == 0)
                                        {
                                                cstate.d = -cstate.g;
                                        }
                                        else
                                        {
                                                const tscalar beta = cstate.g.dot(cstate.g - pstate.g) /
                                                                      pstate.g.dot(pstate.g);
                                                cstate.d = -cstate.g + std::max(static_cast<tscalar>(0), beta) * pstate.d;
                                        }

                                        // update solution
                                        const tscalar t = ls_strong_wolfe(problem, cstate, ft, gt, 1e-4, 0.1, 64);
                                        if (t < std::numeric_limits<tscalar>::epsilon())
                                        {
                                                log_warning() << "optimize: line-search failed for CGD!";
                                                break;
                                        }
                                        pstate = cstate;
                                        cstate.update(t, ft, gt);
                                }

                                return result;
                        }

                        /////////////////////////////////////////////////////////////////////////////////////////////
                        // limited memory bfgs (l-bfgs) starting from the initial value (guess) x0.
                        /////////////////////////////////////////////////////////////////////////////////////////////

                        static tresult _lbfgs(
                                const tproblem& problem,
                                const tvector& x0,
                                tsize max_iterations,          // maximum number of iterations
                                tscalar epsilon,               // convergence precision
                                tsize history_size = 8,        // hessian approximation history size
                                const op_updated_t& op_updated = op_updated_t())
                        {
                                assert(problem.size() == math::cast<tsize>(x0.size()));

                                tresult result(problem.size());
                                std::deque<tvector> ss, ys;
                                tstate cstate(problem, x0), pstate = cstate;

                                tvector q, r;
                                tscalar ft;
                                tvector gt;

                                // iterate until convergence
                                for (tsize i = 0; i < max_iterations; i ++)
                                {
                                        result.update(problem, cstate);
                                        if (op_updated)
                                        {
                                                op_updated(result);
                                        }

                                        // check convergence
                                        if (cstate.converged(epsilon))
                                        {
                                                break;
                                        }

                                        // descent direction
                                        //      (LBFGS - Nocedal & Wright (numerical optimization 2nd) notations @ p.178)
                                        q = cstate.g;

                                        typename std::deque<tvector>::const_reverse_iterator itr_s = ss.rbegin();
                                        typename std::deque<tvector>::const_reverse_iterator itr_y = ys.rbegin();
                                        std::vector<tscalar> alphas;
                                        for (tsize j = 1; j <= history_size && i >= j; j ++)
                                        {
                                                const tvector& s = (*itr_s ++);
                                                const tvector& y = (*itr_y ++);

                                                const tscalar alpha = s.dot(q) / s.dot(y);
                                                q.noalias() -= alpha * y;
                                                alphas.push_back(alpha);
                                        }

                                        if (i == 0)
                                        {
                                                r = q;
                                        }
                                        else
                                        {
                                                const tvector& s = *ss.rbegin();
                                                const tvector& y = *ys.rbegin();
                                                r = s.dot(y) / y.dot(y) * q;
                                        }

                                        typename std::deque<tvector>::const_iterator it_s = ss.begin();
                                        typename std::deque<tvector>::const_iterator it_y = ys.begin();
                                        typename std::vector<tscalar>::const_reverse_iterator itr_alpha = alphas.rbegin();
                                        for (tsize j = 1; j <= history_size && i >= j; j ++)
                                        {
                                                const tvector& s = (*it_s ++);
                                                const tvector& y = (*it_y ++);

                                                const tscalar alpha = *(itr_alpha ++);
                                                const tscalar beta = y.dot(r) / s.dot(y);
                                                r.noalias() += s * (alpha - beta);
                                        }

                                        cstate.d = -r;

                                        // update solution
                                        const tscalar t = ls_strong_wolfe(problem, cstate, ft, gt, 1e-4, 0.9);
                                        if (t < std::numeric_limits<tscalar>::epsilon())
                                        {
                                                log_warning() << "optimize: line-search failed for LBFGS!";
                                                break;
                                        }
                                        pstate = cstate;
                                        cstate.update(t, ft, gt);

                                        ss.push_back(cstate.x - pstate.x);
                                        ys.push_back(cstate.g - pstate.g);
                                        if (ss.size() > history_size)
                                        {
                                                ss.pop_front();
                                                ys.pop_front();
                                        }
                                }

                                return result;
                        }

                        //-------------------------------------------------------------------------------------------------

                        // check and force a descent direction

                        static tscalar descent(tstate& st)
                        {
                                tscalar dg = st.d.dot(st.g);
                                if (dg > std::numeric_limits<tscalar>::min())
                                {
                                        log_warning() << "optimize: not a descent direction!";
                                        st.d = -st.g;
                                        dg = st.d.dot(st.g);
                                }

                                return dg;
                        }

                        /////////////////////////////////////////////////////////////////////////////////////////////
                        // line-search methods to find the scalar that reduces
                        //      the function value (the most) along the direction d:
                        //
                        //      argmin(t) f(x + t * d).
                        /////////////////////////////////////////////////////////////////////////////////////////////

                        // Armijo (sufficient decrease) condition
                        static tscalar ls_armijo(const tproblem& problem, tstate& st,
                                tscalar alpha = 0.2, tscalar beta = 0.7, tsize max_iters = 64)
                        {
                                const tscalar dg = descent(st);

                                tscalar t = 1;
                                for (tsize i = 0; i < max_iters; i ++, t = beta * t)
                                {
                                        if (problem.f(st.x + t * st.d) < st.f + t * alpha * dg)
                                        {
                                                return t;
                                        }
                                }

                                return 0;
                        }

                        // helper function: strong Wolfe (sufficient decrease and curvature) conditions
                        static tscalar ls_zoom(const tproblem& problem, const tstate& st,
                                tscalar& ft, tvector& gt,
                                tscalar tlo, tscalar thi, tscalar ftlo, tscalar fthi,
                                tscalar c1, tscalar c2, tsize max_iters = 64)
                        {
                                const tscalar dg = st.d.dot(st.g);

                                // (Nocedal & Wright (numerical optimization 2nd) @ p.60)
                                for (tsize i = 0; i < max_iters; i ++)
                                {
                                        const tscalar t = (tlo + thi) / 2;

                                        // check sufficient decrease
                                        ft = problem.f(st.x + t * st.d, gt);
                                        if (ft > st.f + c1 * t * dg || ft >= ftlo)
                                        {
                                                thi = t;
                                                fthi = ft;
                                        }

                                        // check curvature
                                        else
                                        {
                                                const tscalar dg1 = gt.dot(st.d);
                                                if (std::fabs(dg1) <= -c2 * dg)
                                                {
                                                        return t;
                                                }

                                                if (dg1 * (thi - tlo) >= 0)
                                                {
                                                        thi = tlo;
                                                        fthi = ftlo;
                                                }

                                                tlo = t;
                                                ftlo = ft;
                                        }
                                }

                                return 0;
                        }

                        // strong Wolfe (sufficient decrease and curvature) conditions
                        static tscalar ls_strong_wolfe(const tproblem& problem, tstate& st,
                                tscalar& ft, tvector& gt,
                                tscalar c1 = 1e-4, tscalar c2 = 0.1, tsize max_iters = 64)
                        {
                                const tscalar dg = descent(st);

                                const tscalar tmax = 1000;

                                tscalar t0 = 0, ft0 = std::numeric_limits<tscalar>::max();
                                tscalar t = 1;

                                // (Nocedal & Wright (numerical optimization 2nd) @ p.60)
                                for (tsize i = 0; i < max_iters; i ++)
                                {
                                        // check sufficient decrease
                                        ft = problem.f(st.x + t * st.d, gt);
                                        if (ft > st.f + c1 * t * dg || ft >= ft0)
                                        {
                                                return ls_zoom(problem, st, ft, gt, t0, t, ft0, ft, c1, c2, max_iters);
                                        }

                                        // check curvature
                                        const tscalar dg1 = gt.dot(st.d);
                                        if (std::fabs(dg1) <= -c2 * dg)
                                        {
                                                return t;
                                        }

                                        if (dg1 >= 0)
                                        {
                                                return ls_zoom(problem, st, ft, gt, t, t0, ft, ft0, c1, c2, max_iters);
                                        }

                                        t0 = t;
                                        t = std::min(tmax, t * 3);
                                        ft0 = ft;
                                }

                                return 0;
                        }
                };
        }
}

#endif // NANOCV_OPTIMIZE_ALGO_HPP
