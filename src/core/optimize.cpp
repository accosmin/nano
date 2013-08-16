#include "optimize.h"
#include "cast.h"
#include "logger.h"
#include <deque>
#include <cassert>
#include <map>

namespace ncv
{
        namespace optimize
        {
                /////////////////////////////////////////////////////////////////////////////////////////////
                // checks the convergence:
                //      the gradient is relatively small.
                /////////////////////////////////////////////////////////////////////////////////////////////

                namespace impl
                {
                        inline bool converged(scalar_t epsilon, const state_t& st)
                        {
                                return st.g.lpNorm<Eigen::Infinity>() < epsilon * (1.0 + std::fabs(st.f));
                        }
                }

                /////////////////////////////////////////////////////////////////////////////////////////////
                // backtracking line-search searches for the scalar
                // that reduces the function value (the most) along the direction d:
                //
                //      argmin(t) f(x + t * d).
                /////////////////////////////////////////////////////////////////////////////////////////////

                namespace impl
                {
                        scalar_t ls_armijo(
                                const problem_t& problem, state_t& st, scalar_t t0,
                                scalar_t alpha = 0.2, scalar_t beta = 0.7)
                        {
                                // Check if descent direction
                                scalar_t dg = st.d.dot(st.g);
                                if (dg > std::numeric_limits<scalar_t>::min())
                                {
                                        st.d = -st.g;
                                        dg = st.d.dot(st.g);
                                }

                                // Armijo (sufficient decrease) condition
                                scalar_t t = t0;
                                while (problem.f(st.x + t * st.d) > st.f + t * alpha * dg)
                                {
                                        t = beta * t;
                                }

                                return t;
                        }

                        scalar_t ls_strong_wolfe(
                                const problem_t& problem, state_t& st, scalar_t t0,
                                scalar_t& ft, vector_t& gt,
                                scalar_t c1 = 1e-4, scalar_t c2 = 0.9)
                        {
                                // Check if descent direction
                                scalar_t dg = st.d.dot(st.g);
                                if (dg > std::numeric_limits<scalar_t>::min())
                                {
                                        st.d = -st.g;
                                        dg = st.d.dot(st.g);
                                }

                                // strong Wolfe (sufficient decrease and curvature) conditions
                                scalar_t t = t0;
                                while ((ft = problem.f(st.x + t * st.d, gt)) > st.f + t * c1 * dg ||
                                       std::fabs(gt.dot(st.d)) < c2 * dg)
                                {
                                        t = c2 * t;
                                }

                                return t;
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        optimize::result_t::result_t(size_t size)
                :       m_optimum(size),
                        m_iterations(0)
        {
        }

        //-------------------------------------------------------------------------------------------------

        void optimize::result_t::update(const state_t& st)
        {
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

        optimize::problem_t::problem_t(
                const op_size_t& op_size,
                const op_fval_t& op_fval,
                const op_fval_grad_t& op_fval_grad)
                :       m_op_size(op_size),
                        m_op_fval(op_fval),
                        m_op_fval_grad(op_fval_grad)
        {
        }

        //-------------------------------------------------------------------------------------------------

        optimize::problem_t::problem_t(
                const op_size_t& op_size,
                const op_fval_t& op_fval)
                :       m_op_size(op_size),
                        m_op_fval(op_fval),
                        m_op_fval_grad()
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
                return m_op_fval(x);
        }

        //-------------------------------------------------------------------------------------------------

        scalar_t optimize::problem_t::f(const vector_t x, vector_t& g) const
        {
                if (m_op_fval_grad)
                {
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
                scalar_t t = 1.0, pdt = -1.0;

                // iterate until convergence
                for (size_t i = 0; i < max_iterations; i ++)
                {
                        result.update(cstate);
                        if (op_updated)
                        {
                                op_updated(result);
                        }

                        // check convergence
                        if (impl::converged(epsilon, cstate))
                        {
                                break;
                        }

                        // descent direction
                        cstate.d = -cstate.g;

                        // update solution
                        const scalar_t dt = cstate.g.dot(cstate.d);
                        if (i > 0)
                        {
                                t *= pdt / dt;
                        }

                        t = impl::ls_armijo(problem, cstate, t);
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
                        result.update(cstate);
                        if (op_updated)
                        {
                                op_updated(result);
                        }

                        // check convergence
                        if (impl::converged(epsilon, cstate))
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
                                cstate.d = -cstate.g +
                                           std::max(static_cast<scalar_t>(0.0), beta) * pstate.d;
                        }

                        // update solution
                        const scalar_t t = impl::ls_strong_wolfe(problem, cstate, 1.0, ft, gt);
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
                        result.update(cstate);
                        if (op_updated)
                        {
                                op_updated(result);
                        }

                        // check convergence
                        if (impl::converged(epsilon, cstate))
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
                        const scalar_t t = impl::ls_strong_wolfe(problem, cstate, 1.0, ft, gt);
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

        namespace optimize
        {
                /////////////////////////////////////////////////////////////////////////////////////////////
                // run stochastic gradient iterations using the given learning rate.
                //      update the function value each <update_iterations>.
                /////////////////////////////////////////////////////////////////////////////////////////////

                namespace impl
                {
                        result_t sgd(
                                const problem_t& problem, const state_t& state,
                                scalar_t eta, scalar_t lambda, scalar_t epsilon,
                                size_t opt_iters, size_t update_iters, const op_updated_t& op_updated,
                                bool average)
                        {
                                result_t result(problem.size());
                                state_t cstate = state;

                                vector_t avg_x;
                                if (average)
                                {
                                        avg_x.resize(problem.size());
                                        avg_x.setZero();
                                }

                                // optimize ...
                                for (size_t t = 0; t < opt_iters; t ++)
                                {
                                        const scalar_t learning_rate = eta / (1.0 + eta * lambda * t);

                                        problem.f(cstate.x, cstate.g);
                                        cstate.x.noalias() -= learning_rate * cstate.g;

                                        // average sgd
                                        if (average)
                                        {
                                                avg_x = ((t + 0.0) * avg_x + cstate.x) / (t + 1.0);
                                        }

                                        // check convergence (& update function value)
                                        if (impl::converged(epsilon, cstate))
                                        {
                                                cstate.f = problem.f(cstate.x);
                                                result.update(cstate);

                                                if (op_updated)
                                                {
                                                        op_updated(result);
                                                }

                                                break;
                                        }

                                        // update function value (from time to time)
                                        if (    ((t + 1) % update_iters) == 0 ||
                                                (t + 1) == opt_iters)
                                        {
                                                cstate.f = problem.f(cstate.x);
                                                result.update(cstate);

                                                if (op_updated)
                                                {
                                                        op_updated(result);
                                                }
                                        }
                                }

                                // average sgd
                                if (average)
                                {
                                        cstate.x = avg_x;
                                        cstate.f = problem.f(cstate.x);
                                        result.update(cstate);
                                }

                                return result;
                        }
                }

                /////////////////////////////////////////////////////////////////////////////////////////////
                // run stochastic gradient iterations & optimize (tune) the learning rate.
                /////////////////////////////////////////////////////////////////////////////////////////////

                namespace impl
                {
                        result_t sgd(
                                const problem_t& problem, const vector_t& x0,
                                size_t iterations, scalar_t epsilon, const op_updated_t& op_updated,
                                bool average)
                        {
                                assert(problem.size() == math::cast<size_t>(x0.size()));

                                const size_t opt_iters = std::max(static_cast<size_t>(1), iterations);
                                const size_t tune_iters = std::max(static_cast<size_t>(1), iterations / 20);
                                const size_t update_iters = std::max(static_cast<size_t>(1), iterations / 10);

                                state_t cstate(problem, x0);

                                // tune the learning rate
                                const scalar_t lambda = 1.0;
                                const scalars_t etas = { 1e-5, 1e-4, 1e-3, 1e-2, 1e-1 };

                                std::map<scalar_t, scalar_t> eta_results;
                                for (scalar_t eta : etas)
                                {
                                        log_info() << (average ? "ASGD" : "SGD")
                                                   << ": learning rate [eta = " << eta << "] ...";

                                        const result_t res = sgd(problem, cstate, eta, lambda, epsilon,
                                                                 tune_iters, tune_iters, op_updated, average);

                                        eta_results[res.optimum().f] = eta;
                                }

                                const scalar_t eta = eta_results.begin()->second;
                                log_info() << (average ? "ASGD" : "SGD")
                                           << ": optimum learning rate [eta = " << eta << "].";

                                // optimize using the optimal learning rate
                                return sgd(problem, cstate, eta, lambda, epsilon,
                                           opt_iters, update_iters, op_updated, average);
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        optimize::result_t optimize::sgd(
                const problem_t& problem, const vector_t& x0, size_t iterations, scalar_t epsilon,
                const op_updated_t& op_updated)
        {
                return impl::sgd(problem, x0, iterations, epsilon, op_updated, false);
        }

        //-------------------------------------------------------------------------------------------------

        optimize::result_t optimize::asgd(
                const problem_t& problem, const vector_t& x0, size_t iterations, scalar_t epsilon,
                const op_updated_t& op_updated)
        {
                return impl::sgd(problem, x0, iterations, epsilon, op_updated, true);
        }

        //-------------------------------------------------------------------------------------------------
}
