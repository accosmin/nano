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
                        inline bool converged(const vector_t& g, scalar_t f, scalar_t eps)
                        {
                                return g.lpNorm<Eigen::Infinity>() < eps * (1.0 + std::fabs(f));
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
                                const tproblem& problem,
                                const vector_t& x,
                                scalar_t t0,
                                const vector_t& d,
                                const scalar_t f,
                                const vector_t& g,
                                scalar_t alpha = 0.2, scalar_t beta = 0.7)
                        {
                                const scalar_t dg = d.dot(g);

                                // Armijo (sufficient decrease) condition
                                scalar_t t = t0;
                                while (problem.f(x + t * d) > f + t * alpha * dg)
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
                                const tproblem& problem,
                                const vector_t& x,
                                scalar_t t0,
                                const vector_t& d,
                                const scalar_t f,
                                const vector_t& g,
                                scalar_t c1 = 1e-4, scalar_t c2 = 0.1)
                        {
                                const scalar_t dg = d.dot(g);
                                vector_t gt;

                                // strong Wolfe (sufficient decrease and curvature) conditions
                                scalar_t t = t0;
                                while (problem.f(x + t * d, gt) > f + t * c1 * dg ||
                                       std::fabs(gt.dot(d)) < c2 * dg)
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

                        vector_t x = x0, g, pg, d;
                        scalar_t t = 1.0, dt = -1.0, pdt = -1.0;

                        // iterate until convergence
                        for (index_t i = 0; i < problem.max_iterations(); i ++, pg = g, pdt = dt)
                        {
                                const scalar_t f = problem.f(x, g);
                                problem.update(x, f, g.norm());

                                // check convergence
                                if (impl::converged(g, f, problem.epsilon()))
                                {
                                        return true;
                                }

                                // descent direction
                                d = -g;

                                // update solution
                                dt = g.dot(d);
                                if (i > 0)
                                {
                                        t *= pdt / dt;
                                }

                                t = impl::line_search_armijo(problem, x, t, d, f, g);
                                x.noalias() += t * d;
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

                        vector_t x = x0, g, pg, d, pd;
                        scalar_t t = 1.0, dt = -1.0, pdt = -1.0;

                        // iterate until convergence
                        for (index_t i = 0; i < problem.max_iterations(); i ++, pg = g, pd = d, pdt = dt)
                        {
                                const scalar_t f = problem.f(x, g);
                                problem.update(x, f, g.norm());

                                // check convergence
                                if (impl::converged(g, f, problem.epsilon()))
                                {
                                        return true;
                                }

                                // descent direction (Polak–Ribière updates)
                                if (i == 0)
                                {
                                        d = -g;
                                }
                                else
                                {
                                        const scalar_t beta = g.dot(g - pg) / (pg.dot(pg));
                                        d = -g + std::max(static_cast<scalar_t>(0.0), beta) * pd;
                                }

                                if (d.dot(g) > std::numeric_limits<scalar_t>::min())
                                {
                                        d = -g;
                                }

                                // update solution
                                dt = g.dot(d);
                                if (i > 0)
                                {
                                        t *= pdt / dt;
                                }

                                t = impl::line_search_strong_wolfe(problem, x, t, d, f, g);
                                x.noalias() += t * d;
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
                        size_t history_size = 6)
                {
                        if (problem.size() != math::cast<size_t>(x0.size()))
                        {
                                return false;
                        }

                        problem.clear();

                        vector_t x = x0, px, g, pg, d, pd;
                        scalar_t t = 1.0;
                        matrix_t dxs(problem.size(), history_size);
                        matrix_t dgs(problem.size(), history_size);

                        // iterate until convergence
                        for (index_t i = 0; i < problem.max_iterations(); i ++, px = x, pg = g, pd = d)
                        {
                                const scalar_t f = problem.f(x, g);
                                problem.update(x, f, g.norm());

                                // check convergence
                                if (impl::converged(g, f, problem.epsilon()))
                                {
                                        return true;
                                }

                                // descent direction (LBFGS)
                                if (i == 0)
                                {
                                        d = -g;
                                }
                                else
                                {
                                        d = -g;
                                }

                                // update solution
                                t = impl::line_search_strong_wolfe(problem, x, 1.0, d, f, g);
                                x.noalias() += t * d;
                        }

                        return false;
                }
        }
}

#endif // NANOCV_OPTIMIZE_H
