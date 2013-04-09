#ifndef NANOCV_OPTIMIZE_H
#define NANOCV_OPTIMIZE_H

#include "ncv_math.h"

namespace ncv
{
        namespace optimize
        {
                /////////////////////////////////////////////////////////////////////////////////////////////
                // describes a multivariate optimization problem.
                /////////////////////////////////////////////////////////////////////////////////////////////

                template
                <
                        typename tscalar,               // input type
                        typename top_size,              // dimensionality:              N = top_size()
                        typename top_fval,              // function value:              fx = top_fval(x)
                        typename top_fval_grad          //  & gradient:                 fx = top_fval_grad(x, gx)
                >
                class problem
                {
                public:

                        typedef tscalar                                 scalar_t;
                        typedef typename vector<scalar_t>::vector_t     vector_t;
                        typedef typename vector<scalar_t>::vectors_t    vectors_t;

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
                        void update(const vector_t& x, scalar_t fx, scalar_t gn) const
                        {
                                m_iterations ++;
                                m_opt_x = x;
                                m_opt_fx = fx;
                                m_opt_gn = gn;
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
                };

                /////////////////////////////////////////////////////////////////////////////////////////////
                // checks the convergence:
                //      the gradient is relatively small.
                /////////////////////////////////////////////////////////////////////////////////////////////

                namespace impl
                {
                        template
                        <
                                typename tvector,
                                typename tscalar
                        >
                        bool converged(const tvector& g, const tvector& g_prv, tscalar eps)
                        {
                                if (g.size() != g_prv.size())
                                {
                                        return  math::cast<tscalar>(g.norm()) < eps;
                                }
                                else
                                {
                                        return  math::cast<tscalar>(g.norm()) < eps ||
                                                math::cast<tscalar>((g - g_prv).norm()) < eps;
                                }
                        }
                }

                /////////////////////////////////////////////////////////////////////////////////////////////
                // backtracking line-search searches for the scalar
                // that reduces the function value (the most) along the direction d:
                //
                //      argmin(t) f(x + t * d).
                //
                // NB: the direction is set automatically to -gradient if it is not a descent direction.
                /////////////////////////////////////////////////////////////////////////////////////////////

                namespace impl
                {
                        template
                        <
                                typename tproblem
                        >
                        typename tproblem::scalar_t line_search(
                                const tproblem& problem,
                                const typename tproblem::vector_t& x,
                                typename tproblem::scalar_t t0,                 // Initial step size
                                typename tproblem::vector_t& d,
                                const typename tproblem::scalar_t& fx,
                                const typename tproblem::vector_t& gx,
                                typename tproblem::scalar_t alpha = math::cast<typename tproblem::scalar_t>(0.2),
                                typename tproblem::scalar_t beta = math::cast<typename tproblem::scalar_t>(0.7))
                        {
                                typedef typename tproblem::scalar_t     scalar_t;

                                // Reset to gradient descent if the current direction is not
                                if (-d.dot(gx) < 2.0 * problem.epsilon())
                                {
                                        d = -gx;
                                }

                                const scalar_t f0 = fx;
                                const scalar_t d0 = d.dot(gx);
                                t0 = math::clamp(t0, 1e-6, 1.0);

                                const index_t max_steps = 8;
                                const index_t max_trials = 64;

                                // Try various sufficient decrease steps ...
                                for (index_t step = 0; step < max_steps; step ++, alpha *= 0.1)
                                {
                                        index_t trials = 0;

                                        // Armijo (sufficient decrease) condition
                                        scalar_t t = t0;
                                        while ((++ trials) < max_trials &&
                                               problem.f(x + t * d) > f0 + t * alpha * d0)
                                        {
                                                t = beta * t;
                                        }

                                        if (trials < max_trials)
                                        {
                                                return t;
                                        }
                                }

                                return 0.0;
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
                        const typename tproblem::vector_t& x0)
                {
                        typedef typename tproblem::scalar_t     scalar_t;
                        typedef typename tproblem::vector_t     vector_t;

                        if (problem.size() != math::cast<size_t>(x0.size()))
                        {
                                return false;
                        }

                        problem.clear();

                        vector_t x = x0, gx, gx_prv, d;
                        scalar_t t = 1.0, dt = 1.0, dt_prv = 1.0;

                        // iterate until convergence
                        for (index_t i = 0; i < problem.max_iterations(); i ++, gx_prv = gx, dt_prv = dt)
                        {
                                const scalar_t fx = problem.f(x, gx);
                                problem.update(x, fx, gx.norm());

                                // check convergence
                                if (impl::converged(gx, gx_prv, problem.epsilon()))
                                {
                                        break;
                                }

                                // descent direction
                                d = -gx;

                                // update solution
                                dt = gx.norm();
                                if (i > 0)
                                {
                                        t *= dt_prv / dt;
                                }

                                t = impl::line_search(problem, x, t, d, fx, gx, 0.2, 0.7);
                                x.noalias() += t * d;
                        }

                        return true;
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
                        const typename tproblem::vector_t& x0)
                {
                        typedef typename tproblem::scalar_t     scalar_t;
                        typedef typename tproblem::vector_t     vector_t;

                        if (problem.size() != math::cast<size_t>(x0.size()))
                        {
                                return false;
                        }

                        problem.clear();

                        vector_t x = x0, gx, gx_prv, d, d_prv;
                        scalar_t t = 1.0, dt = 1.0, dt_prv = 1.0;

                        // iterate until convergence
                        for (index_t i = 0; i < problem.max_iterations(); i ++, gx_prv = gx, d_prv = d, dt_prv = dt)
                        {
                                const scalar_t fx = problem.f(x, gx);
                                problem.update(x, fx, gx.norm());

                                // check convergence
                                if (impl::converged(gx, gx_prv, problem.epsilon()))
                                {
                                        break;
                                }

                                // descent direction (Polak–Ribière updates)
                                if (i == 0)
                                {
                                        d = -gx;
                                }
                                else
                                {
                                        const scalar_t beta = gx.dot(gx - gx_prv) / (gx_prv.dot(gx_prv));
                                        d = -gx + std::max(static_cast<scalar_t>(0.0), beta) * d_prv;
                                }

                                // update solution
                                dt = gx.norm();
                                if (i > 0)
                                {
                                        t *= dt_prv / dt;
                                }

                                t = impl::line_search(problem, x, t, d, fx, gx, 0.2, 0.7);
                                x.noalias() += t * d;
                        }

                        return true;
                }
        }
}

#endif // NANOCV_OPTIMIZE_H
