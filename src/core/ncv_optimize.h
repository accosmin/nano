#ifndef NANOCV_OPTIMIZE_H
#define NANOCV_OPTIMIZE_H

#include "ncv_math.h"
#include "ncv_stats.h"
#include <deque>

namespace ncv
{
        namespace optimize
        {
                /////////////////////////////////////////////////////////////////////////////////////////////
                // optimization state: current point, function value, gradient and descent direction.
                ////////////////////////////////////////////////////////////////////////////////////////////////

                struct state
                {
                        // constructor
                        template <class tproblem>
                        state(const tproblem& problem, const vector_t& x0)
                        {
                                x = x0;
                                f = problem.f(x, g);
                        }

                        // update current point
                        template <class tproblem>
                        void update(const tproblem& problem, scalar_t t)
                        {
                                x.noalias() += t * d;
                                f = problem.f(x, g);
                        }

                        // attributes
                        vector_t x, g, d;
                        scalar_t f;
                };

                /////////////////////////////////////////////////////////////////////////////////////////////
                // describes a multivariate optimization problem.
                /////////////////////////////////////////////////////////////////////////////////////////////

                template
                <
                        typename top_size,              // dimensionality:              N = top_size()
                        typename top_fval,              // function value:              fx = top_fval(x)
                        typename top_fval_grad          //  & gradient:                 fx = top_fval_grad(x, gx)
                >
                class problem
                {
                public:

                        // constructor
                        problem(const top_size& op_size,
                                const top_fval& op_fval,
                                const top_fval_grad& op_fval_grad,
                                size_t max_iterations,  // maximum number of iterations (stopping criteria)
                                scalar_t epsilon)       // desired precision (stopping criteria)
                                :       m_max_iterations(max_iterations),
                                        m_epsilon(epsilon),
                                        m_op_size(op_size),
                                        m_op_fval(op_fval),
                                        m_op_fval_grad(op_fval_grad),
                                        m_has_op_grad(true)
                        {
                                clear();
                        }

                        // constructor
                        problem(const top_size& op_size,
                                const top_fval& op_fval,
                                // no gradient provided!
                                size_t max_iterations,  // maximum number of iterations (stopping criteria)
                                scalar_t epsilon)       // desired precision (stopping criteria)
                                :       m_max_iterations(max_iterations),
                                        m_epsilon(epsilon),
                                        m_op_size(op_size),
                                        m_op_fval(op_fval),
                                        m_has_op_grad(false)
                        {
                                clear();
                        }

                        // clear history
                        void clear() const
                        {
                                m_opt_x.resize(size()); m_opt_x.setZero();
                                m_opt_fx = std::numeric_limits<scalar_t>::max();
                                m_opt_gn = std::numeric_limits<scalar_t>::max();
                                m_f_evals = 0;
                                m_g_evals = 0;
                                m_iterations = 0;
                                m_speed_stats.clear();
                        }

                        // compute function value & gradient
                        scalar_t f(const vector_t& x) const
                        {
                                m_f_evals ++;
                                return m_op_fval(x);
                        }
                        scalar_t f(const vector_t x, vector_t& g) const
                        {
                                if (m_has_op_grad)
                                {
                                        m_f_evals ++;
                                        m_g_evals ++;
                                        return m_op_fval_grad(x, g);
                                }
                                else
                                {
                                        eval_grad(x, g);
                                        return f(x);
                                }
                        }

                        // update optimal
                        void update(const state& st) const
                        {
                                if (m_iterations > 0)
                                {
                                        const scalar_t df = std::fabs(m_opt_fx - st.f);
                                        m_speed_stats.add(df / std::max(1.0, std::fabs(m_opt_fx)));
                                }

                                m_iterations ++;
                                if (st.f < m_opt_fx)
                                {
                                        m_opt_x = st.x;
                                        m_opt_fx = st.f;
                                        m_opt_gn = st.g.norm();
                                }
                        }

                        // access functions
                        size_t size() const { return m_op_size(); }

                        size_t max_iterations() const { return m_max_iterations; }
                        scalar_t epsilon() const { return m_epsilon; }

                        const vector_t& opt_x() const { return m_opt_x; }
                        scalar_t opt_fx() const { return m_opt_fx; }
                        scalar_t opt_gn() const { return m_opt_gn; }

                        count_t fevals() const { return m_f_evals; }
                        count_t gevals() const { return m_g_evals; }
                        count_t iterations() const { return m_iterations; }
                        scalar_t speed_avg() const { return m_speed_stats.avg(); }
                        scalar_t speed_stdev() const { return m_speed_stats.stdev(); }

                private:

                        // evaluate gradient (if not provided)
                        void eval_grad(const vector_t x, vector_t& g) const
                        {
                                const size_t n = size();
                                const scalar_t d = epsilon();

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

                private:

                        // attributes
                        size_t                  m_max_iterations;
                        scalar_t                m_epsilon;

                        top_size                m_op_size;
                        top_fval                m_op_fval;
                        top_fval_grad           m_op_fval_grad;
                        bool                    m_has_op_grad;

                        mutable vector_t        m_opt_x;                // optimal value
                        mutable scalar_t        m_opt_fx;               // optimal function value
                        mutable scalar_t        m_opt_gn;               // optimal function gradient norm

                        mutable count_t         m_f_evals;              // #function value evaluations
                        mutable count_t         m_g_evals;              // #function gradient evaluations
                        mutable count_t         m_iterations;           // #iterations
                        mutable stats<scalar_t> m_speed_stats;          // convergence speed statistics
                };

                /////////////////////////////////////////////////////////////////////////////////////////////
                // checks the convergence:
                //      the gradient is relatively small.
                /////////////////////////////////////////////////////////////////////////////////////////////

                namespace impl
                {
                        template
                        <
                                class tproblem
                        >
                        inline bool converged(const tproblem& problem, const state& st)
                        {
                                return st.g.lpNorm<Eigen::Infinity>() < problem.epsilon() * (1.0 + std::fabs(st.f));
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
                        template
                        <
                                typename tproblem
                        >
                        scalar_t line_search_armijo(
                                const tproblem& problem, state& st, scalar_t t0,
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

                        template
                        <
                                typename tproblem
                        >
                        scalar_t line_search_strong_wolfe(
                                const tproblem& problem, state& st, scalar_t t0,
                                scalar_t c1 = 1e-4, scalar_t c2 = 0.1)
                        {
                                // Check if descent direction
                                scalar_t dg = st.d.dot(st.g);
                                if (dg > std::numeric_limits<scalar_t>::min())
                                {
                                        st.d = -st.g;
                                        dg = st.d.dot(st.g);
                                }

                                vector_t gt;

                                // strong Wolfe (sufficient decrease and curvature) conditions
                                scalar_t t = t0;
                                while (problem.f(st.x + t * st.d, gt) > st.f + t * c1 * dg ||
                                       std::fabs(gt.dot(st.d)) < c2 * dg)
                                {
                                        t = c2 * t;
                                }

                                return t;
                        }
                }

                /////////////////////////////////////////////////////////////////////////////////////////////
                // gradient descent starting from the initial value (guess) x0.
                /////////////////////////////////////////////////////////////////////////////////////////////

                template
                <
                        class tproblem
                >
                bool gradient_descent(
                        const tproblem& problem,
                        const vector_t& x0)
                {
                        if (problem.size() != math::cast<size_t>(x0.size()))
                        {
                                return false;
                        }

                        problem.clear();

                        state cstate(problem, x0);
                        scalar_t t = 1.0, dt = -1.0, pdt = -1.0;

                        // iterate until convergence
                        for (index_t i = 0; i < problem.max_iterations(); i ++)
                        {
                                problem.update(cstate);

                                // check convergence
                                if (impl::converged(problem, cstate))
                                {
                                        return true;
                                }

                                // descent direction
                                cstate.d = -cstate.g;

                                // update solution
                                dt = cstate.g.dot(cstate.d);
                                if (i > 0)
                                {
                                        t *= pdt / dt;
                                }

                                t = impl::line_search_armijo(problem, cstate, t, 0.2, 0.7);
                                cstate.update(problem, t);
                        }                        

                        return false;
                }

                /////////////////////////////////////////////////////////////////////////////////////////////
                // conjugate gradient descent starting from the initial value (guess) x0.
                /////////////////////////////////////////////////////////////////////////////////////////////

                template
                <
                        class tproblem
                >
                bool conjugate_gradient_descent(
                        const tproblem& problem,
                        const vector_t& x0)
                {
                        if (problem.size() != math::cast<size_t>(x0.size()))
                        {
                                return false;
                        }

                        problem.clear();

                        state cstate(problem, x0), pstate = cstate;

                        // iterate until convergence
                        for (index_t i = 0; i < problem.max_iterations(); i ++)
                        {
                                problem.update(cstate);

                                // check convergence
                                if (impl::converged(problem, cstate))
                                {
                                        return true;
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
                                                   std::max(0.0, beta) * pstate.d;
                                }

                                // update solution
                                const scalar_t t = impl::line_search_strong_wolfe(problem, cstate, 1.0, 1e-4, 0.1);
                                pstate = cstate;
                                cstate.update(problem, t);
                        }

                        return false;
                }

                /////////////////////////////////////////////////////////////////////////////////////////////
                // limited memory bfgs (l-bfgs) starting from the initial value (guess) x0.
                /////////////////////////////////////////////////////////////////////////////////////////////

                template
                <
                        class tproblem
                >
                bool lbfgs(
                        const tproblem& problem,
                        const vector_t& x0,
                        size_t history_size = 8)
                {
                        if (problem.size() != math::cast<size_t>(x0.size()))
                        {
                                return false;
                        }

                        problem.clear();

                        std::deque<vector_t> ss, ys;
                        state cstate(problem, x0), pstate = cstate;

                        vector_t q, r;

                        // iterate until convergence
                        for (index_t i = 0; i < problem.max_iterations(); i ++)
                        {
                                problem.update(cstate);

                                // check convergence
                                if (impl::converged(problem, cstate))
                                {
                                        return true;
                                }

                                // descent direction
                                //      (LBFGS - Nocedal & Wright (numerical optimization 2nd) notations @ p.178)
                                q = cstate.g;

                                std::deque<vector_t>::const_reverse_iterator itr_s = ss.rbegin();
                                std::deque<vector_t>::const_reverse_iterator itr_y = ys.rbegin();
                                scalars_t alphas;
                                for (index_t j = 1; j <= history_size && i >= j; j ++)
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
                                for (index_t j = 1; j <= history_size && i >= j; j ++)
                                {
                                        const vector_t& s = (*it_s ++);
                                        const vector_t& y = (*it_y ++);

                                        const scalar_t alpha = *(itr_alpha ++);
                                        const scalar_t beta = y.dot(r) / s.dot(y);
                                        r.noalias() += s * (alpha - beta);
                                }

                                cstate.d = -r;

                                // update solution
                                const scalar_t t = impl::line_search_strong_wolfe(problem, cstate, 1.0, 1e-4, 0.9);
                                pstate = cstate;
                                cstate.update(problem, t);

                                ss.push_back(cstate.x - pstate.x);
                                ys.push_back(cstate.g - pstate.g);
                                if (ss.size() > history_size)
                                {
                                        ss.pop_front();
                                        ys.pop_front();
                                }
                        }

                        return false;
                }
        }
}

#endif // NANOCV_OPTIMIZE_H
