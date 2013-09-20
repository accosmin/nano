#include "optimize.h"
#include "math/cast.hpp"
#include "logger.h"
#include <deque>
#include <cassert>
#include <map>

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        scalar_t optimize::descent(state_t& st)
        {
                scalar_t dg = st.d.dot(st.g);
                if (dg > std::numeric_limits<scalar_t>::min())
                {
                        log_warning() << "optimize: not a descent direction!";
                        st.d = -st.g;
                        dg = st.d.dot(st.g);
                }

                return dg;
        }

        //-------------------------------------------------------------------------------------------------

        scalar_t optimize::ls_armijo(const problem_t& problem, state_t& st,
                scalar_t alpha, scalar_t beta, size_t max_iters)
        {
                const scalar_t dg = descent(st);

                scalar_t t = 1.0;
                for (size_t i = 0; i < max_iters; i ++, t = beta * t)
                {
                        if (problem.f(st.x + t * st.d) < st.f + t * alpha * dg)
                        {
                                return t;
                        }
                }

                return 0.0;
        }

        //-------------------------------------------------------------------------------------------------

        scalar_t optimize::ls_zoom(
                const problem_t& problem, const state_t& st,
                scalar_t& ft, vector_t& gt,
                scalar_t tlo, scalar_t thi, scalar_t ftlo, scalar_t fthi,
                scalar_t c1, scalar_t c2, size_t max_iters)
        {
                const scalar_t dg = st.d.dot(st.g);

                // (Nocedal & Wright (numerical optimization 2nd) @ p.60)
                for (size_t i = 0; i < max_iters; i ++)
                {
                        const scalar_t t = (tlo + thi) / 2.0;

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
                                const scalar_t dg1 = gt.dot(st.d);
                                if (std::fabs(dg1) <= -c2 * dg)
                                {
                                        return t;
                                }

                                if (dg1 * (thi - tlo) >= 0.0)
                                {
                                        thi = tlo;
                                        fthi = ftlo;
                                }

                                tlo = t;
                                ftlo = ft;
                        }
                }

                return 0.0;
        }

        //-------------------------------------------------------------------------------------------------

        scalar_t optimize::ls_strong_wolfe(
                const problem_t& problem, state_t& st, scalar_t& ft, vector_t& gt,
                scalar_t c1, scalar_t c2, size_t max_iters)
        {
                const scalar_t dg = descent(st);

                const scalar_t tmax = 1000.0;

                scalar_t t0 = 0.0, ft0 = std::numeric_limits<scalar_t>::max();
                scalar_t t = 1.0;

                // (Nocedal & Wright (numerical optimization 2nd) @ p.60)
                for (size_t i = 0; i < max_iters; i ++)
                {
                        // check sufficient decrease
                        ft = problem.f(st.x + t * st.d, gt);
                        if (ft > st.f + c1 * t * dg || ft >= ft0)
                        {
                                return ls_zoom(problem, st, ft, gt, t0, t, ft0, ft, c1, c2, max_iters);
                        }

                        // check curvature
                        const scalar_t dg1 = gt.dot(st.d);
                        if (std::fabs(dg1) <= -c2 * dg)
                        {
                                return t;
                        }

                        if (dg1 >= 0.0)
                        {
                                return ls_zoom(problem, st, ft, gt, t, t0, ft, ft0, c1, c2, max_iters);
                        }

                        t0 = t;
                        t = std::min(tmax, t * 3.0);
                        ft0 = ft;
                }

                return 0.0;
        }

        //-------------------------------------------------------------------------------------------------

        optimize::state_t::state_t(size_t size)
                : x(size),
                  g(size),
                  d(size),
                  f(std::numeric_limits<scalar_t>::max()),
                  t(1.0)
        {
        }

        //-------------------------------------------------------------------------------------------------

        optimize::state_t::state_t(const problem_t& problem, const vector_t& x0)
        {
                x = x0;
                f = problem.f(x, g);
        }

        //-------------------------------------------------------------------------------------------------

        void optimize::state_t::update(const problem_t& problem, scalar_t t)
        {
                x.noalias() += t * d;
                f = problem.f(x, g);
        }

        //-------------------------------------------------------------------------------------------------

        void optimize::state_t::update(scalar_t t, scalar_t ft, const vector_t& gt)
        {
                x.noalias() += t * d;
                f = ft;
                g = gt;
        }

        //-------------------------------------------------------------------------------------------------

        optimize::result_t::result_t(size_t size)
                :       m_optimum(size),
                        m_iterations(0),
                        m_cnt_fval(0),
                        m_cnt_grad(0)
        {
        }

        //-------------------------------------------------------------------------------------------------

        void optimize::result_t::update(const problem_t& problem, const state_t& st)
        {
                m_cnt_fval = problem.n_fval_calls();
                m_cnt_grad = problem.n_grad_calls();

                if (m_iterations > 0)
                {
                        const scalar_t df = std::fabs(m_optimum.f - st.f);
                        m_conv_speed.add(df / std::max(static_cast<scalar_t>(1.0), std::fabs(m_optimum.f)));
                }

                m_iterations ++;
                if (st < m_optimum)
                {
                        m_optimum = st;
                }
        }

        //-------------------------------------------------------------------------------------------------

        void optimize::result_t::update(const result_t& result)
        {
                m_cnt_fval += result.m_cnt_fval;
                m_cnt_grad += result.m_cnt_grad;

                m_conv_speed.add(result.m_conv_speed);
                m_iterations += result.m_iterations;

                if (result.m_optimum < m_optimum)
                {
                        m_optimum = result.m_optimum;
                }
        }

        //-------------------------------------------------------------------------------------------------

        optimize::problem_t::problem_t(
                const op_size_t& op_size,
                const op_fval_t& op_fval,
                const op_fval_grad_t& op_fval_grad)
                :       m_op_size(op_size),
                        m_op_fval(op_fval),
                        m_op_fval_grad(op_fval_grad),
                        m_cnt_fval(0),
                        m_cnt_grad(0)
        {
        }

        //-------------------------------------------------------------------------------------------------

        optimize::problem_t::problem_t(
                const op_size_t& op_size,
                const op_fval_t& op_fval)
                :       m_op_size(op_size),
                        m_op_fval(op_fval),
                        m_op_fval_grad(),
                        m_cnt_fval(0),
                        m_cnt_grad(0)
        {
        }

        //-------------------------------------------------------------------------------------------------

        size_t optimize::problem_t::size() const
        {
                return m_op_size();
        }

        //-------------------------------------------------------------------------------------------------

        scalar_t optimize::problem_t::f(const vector_t& x) const
        {
                m_cnt_fval ++;
                return m_op_fval(x);
        }

        //-------------------------------------------------------------------------------------------------

        scalar_t optimize::problem_t::f(const vector_t& x, vector_t& g) const
        {
                if (m_op_fval_grad)
                {
                        m_cnt_fval ++;
                        m_cnt_grad ++;
                        return m_op_fval_grad(x, g);
                }
                else
                {
                        eval_grad(x, g);
                        return f(x);
                }
        }

        //-------------------------------------------------------------------------------------------------

        void optimize::problem_t::eval_grad(const vector_t x, vector_t& g) const
        {
                const size_t n = size();
                const scalar_t d = 1e-6;

                vector_t xp = x, xn = x;

                g.resize(n);
                for (size_t i = 0; i < n; i ++)
                {
                        if (i > 0)
                        {
                                xp(i - 1) -= d;
                                xn(i - 1) += d;
                        }

                        xp(i) += d;
                        xn(i) -= d;
                        g(i) = f(xp) - f(xn);
                }

                g /= 2.0 * d;
        }

        //-------------------------------------------------------------------------------------------------

        optimize::result_t optimize::gd(
                const problem_t& problem, const vector_t& x0,
                size_t max_iterations, scalar_t epsilon,
                const op_updated_t& op_updated)
        {
                assert(problem.size() == math::cast<size_t>(x0.size()));

                result_t result(problem.size());
                state_t cstate(problem, x0);

                // iterate until convergence
                for (size_t i = 0; i < max_iterations; i ++)
                {
                        result.update(problem, cstate);
                        if (op_updated)
                        {
                                op_updated(result);
                        }

                        // check convergence
                        if (optimize::converged(epsilon, cstate))
                        {
                                break;
                        }

                        // descent direction
                        cstate.d = -cstate.g;

                        // update solution
                        const scalar_t t = optimize::ls_armijo(problem, cstate);
                        if (t < std::numeric_limits<scalar_t>::epsilon())
                        {
                                log_warning() << "optimize: line-search failed for GD!";
                                break;
                        }
                        cstate.update(problem, t);
                }

                return result;
        }

        //-------------------------------------------------------------------------------------------------

        optimize::result_t optimize::cgd(
                const problem_t& problem, const vector_t& x0,
                size_t max_iterations, scalar_t epsilon,
                const op_updated_t& op_updated)
        {
                assert(problem.size() == math::cast<size_t>(x0.size()));

                result_t result(problem.size());
                state_t cstate(problem, x0), pstate = cstate;                

                scalar_t ft;
                vector_t gt;

                // iterate until convergence
                for (size_t i = 0; i < max_iterations; i ++)
                {
                        result.update(problem, cstate);
                        if (op_updated)
                        {
                                op_updated(result);
                        }

                        // check convergence
                        if (optimize::converged(epsilon, cstate))
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
                                const scalar_t beta = cstate.g.dot(cstate.g - pstate.g) /
                                                      pstate.g.dot(pstate.g);
                                cstate.d = -cstate.g + std::max(static_cast<scalar_t>(0.0), beta) * pstate.d;
                        }

                        // update solution
                        const scalar_t t = optimize::ls_strong_wolfe(problem, cstate, ft, gt, 1e-4, 0.1, 64);
                        if (t < std::numeric_limits<scalar_t>::epsilon())
                        {
                                log_warning() << "optimize: line-search failed for CGD!";
                                break;
                        }
                        pstate = cstate;
                        cstate.update(t, ft, gt);
                }

                return result;
        }

        //-------------------------------------------------------------------------------------------------

        optimize::result_t optimize::lbfgs(
                const problem_t& problem, const vector_t& x0,
                size_t max_iterations, scalar_t epsilon, size_t history_size,
                const op_updated_t& op_updated)
        {
                assert(problem.size() == math::cast<size_t>(x0.size()));

                result_t result(problem.size());
                std::deque<vector_t> ss, ys;
                state_t cstate(problem, x0), pstate = cstate;

                vector_t q, r;                
                scalar_t ft;
                vector_t gt;

                // iterate until convergence
                for (size_t i = 0; i < max_iterations; i ++)
                {
                        result.update(problem, cstate);
                        if (op_updated)
                        {
                                op_updated(result);
                        }

                        // check convergence
                        if (optimize::converged(epsilon, cstate))
                        {
                                break;
                        }

                        // descent direction
                        //      (LBFGS - Nocedal & Wright (numerical optimization 2nd) notations @ p.178)
                        q = cstate.g;

                        std::deque<vector_t>::const_reverse_iterator itr_s = ss.rbegin();
                        std::deque<vector_t>::const_reverse_iterator itr_y = ys.rbegin();
                        scalars_t alphas;
                        for (size_t j = 1; j <= history_size && i >= j; j ++)
                        {
                                const vector_t& s = (*itr_s ++);
                                const vector_t& y = (*itr_y ++);

                                const scalar_t alpha = s.dot(q) / s.dot(y);
                                q.noalias() -= alpha * y;
                                alphas.push_back(alpha);
                        }

                        if (i == 0)
                        {
                                r = q;
                        }
                        else
                        {
                                const vector_t& s = *ss.rbegin();
                                const vector_t& y = *ys.rbegin();
                                r = s.dot(y) / y.dot(y) * q;
                        }

                        std::deque<vector_t>::const_iterator it_s = ss.begin();
                        std::deque<vector_t>::const_iterator it_y = ys.begin();
                        scalars_t::const_reverse_iterator itr_alpha = alphas.rbegin();
                        for (size_t j = 1; j <= history_size && i >= j; j ++)
                        {
                                const vector_t& s = (*it_s ++);
                                const vector_t& y = (*it_y ++);

                                const scalar_t alpha = *(itr_alpha ++);
                                const scalar_t beta = y.dot(r) / s.dot(y);
                                r.noalias() += s * (alpha - beta);
                        }

                        cstate.d = -r;

                        // update solution
                        const scalar_t t = optimize::ls_strong_wolfe(problem, cstate, ft, gt, 1e-4, 0.9);
                        if (t < std::numeric_limits<scalar_t>::epsilon())
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
}
