#ifndef NANOCV_OPTIMIZE_H
#define NANOCV_OPTIMIZE_H

#include "ncv_math.h"
#include <utility>
#include <eigen3/Eigen/LU>

namespace ncv
{
        namespace optimize
        {
                /////////////////////////////////////////////////////////////////////////////////////////////
                // the set of values & function values & gradient magnitudes evaluated during optimization.
                /////////////////////////////////////////////////////////////////////////////////////////////

                template
                <
                        typename tscalar
                >
                class history
                {
                public:

                        typedef typename vector<tscalar>::vector_t      vector_t;
                        typedef typename vector<tscalar>::vectors_t     vectors_t;

                        typedef typename matrix<tscalar>::matrix_t      matrix_t;
                        typedef typename matrix<tscalar>::matrices_t    matrices_t;

                        // manage history
                        void clear()
                        {
                                m_history.clear();
                        }

                        void memo(const vector_t& x, tscalar fx, tscalar gg)
                        {
                                m_history.push_back(std::make_tuple(x, fx, gg));
                        }

                        // access functions
                        bool empty() const { return m_history.empty(); }
                        size_t size() const { return m_history.size(); }
                        const vector_t& x(index_t i) const { return std::get<0>(m_history[i]); }
                        tscalar fx(index_t i) const { return std::get<1>(m_history[i]); }
                        tscalar gn(index_t i) const { return std::get<2>(m_history[i]); }

                private:

                        // <value = x, function value = f(x), gradient magnitude>
                        typedef std::tuple<vector_t, tscalar, tscalar>  stat_t;
                        typedef std::vector<stat_t>                     stats_t;

                        // attributes
                        stats_t         m_history;
                };

                /////////////////////////////////////////////////////////////////////////////////////////////
                // describes a multivariate optimization problem.
                /////////////////////////////////////////////////////////////////////////////////////////////

                template
                <
                        typename tscalar,               // input type
                        typename top_size,              // dimensionality:              N = top_size()
                        typename top_fval,              // function value:              fx = top_fval(x)
                        typename top_fval_grad,         //  & gradient:                 fx = top_fval_grad(x, gx)
                        typename top_fval_grad_hess     //  & gradient & hessian:       fx = top_fval_grad_hess(x, gx, hx)
                >
                class problem
                {
                public:

                        typedef tscalar                         scalar_t;
                        typedef history<tscalar>                history_t;
                        typedef typename history_t::vector_t    vector_t;
                        typedef typename history_t::vectors_t   vectors_t;
                        typedef typename history_t::matrix_t    matrix_t;
                        typedef typename history_t::matrices_t  matrices_t;

                        // constructor
                        problem(const top_size& op_size,
                                const top_fval& op_fval,
                                const top_fval_grad& op_fval_grad,
                                const top_fval_grad_hess& op_fval_grad_hess,
                                size_t iters,           // maximum number of iterations (stopping criteria)
                                scalar_t eps)           // desired precision (stopping criteria)

                                :       m_iters(iters), m_eps(eps),
                                        m_op_size(op_size),
                                        m_op_fval(op_fval),
                                        m_op_fval_grad(op_fval_grad),
                                        m_op_fval_grad_hess(op_fval_grad_hess)
                        {
                        }

                        // compute function value & gradient & Hessian
                        scalar_t f(const vector_t& x) const
                        {
                                return m_op_fval(x);
                        }
                        scalar_t f(const vector_t x, vector_t& g) const
                        {
                                return m_op_fval_grad(x, g);
                        }
                        scalar_t f(const vector_t x, vector_t& g, matrix_t& h) const
                        {
                                return m_op_fval_grad_hess(x, g, h);
                        }

                        // access functions
                        size_t size() const { return m_op_size(); }
                        size_t iters() const { return m_iters; }
                        scalar_t eps() const { return m_eps; }

                private:

                        // attributes
                        size_t                  m_iters;
                        scalar_t                m_eps;
                        top_size                m_op_size;
                        top_fval                m_op_fval;
                        top_fval_grad           m_op_fval_grad;
                        top_fval_grad_hess      m_op_fval_grad_hess;
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
                        bool converged(const tvector& g, tscalar eps)
                        {
                                const tscalar gnorm = math::cast<tscalar>(g.norm());
                                return gnorm < eps;
                        }
                }

                /////////////////////////////////////////////////////////////////////////////////////////////
                // Backtracking line-search searches for the scalar
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
                        typename tproblem::scalar_t line_search(
                                const tproblem& problem,
                                const typename tproblem::vector_t& x,
                                const typename tproblem::vector_t& d,
                                const typename tproblem::scalar_t& fx,
                                const typename tproblem::vector_t& gx,
                                typename tproblem::scalar_t alpha = math::cast<typename tproblem::scalar_t>(0.2),
                                typename tproblem::scalar_t beta = math::cast<typename tproblem::scalar_t>(0.7))
                        {
                                typedef typename tproblem::scalar_t     scalar_t;

                                const scalar_t f0 = fx;
                                const scalar_t d0 = alpha * gx.dot(d);

                                scalar_t t = math::cast<scalar_t>(1.0);
                                while (problem.f(x + t * d) > f0 + t * d0)
                                {
                                        t = beta * t;
                                }

                                return t;
                        }

                        template
                        <
                                typename tproblem
                        >
                        typename tproblem::scalar_t line_search_curvature(
                                const tproblem& problem,
                                const typename tproblem::vector_t& x,
                                const typename tproblem::vector_t& d,
                                const typename tproblem::scalar_t& fx,
                                const typename tproblem::vector_t& gx,
                                typename tproblem::scalar_t c1 = math::cast<typename tproblem::scalar_t>(1e-4),
                                typename tproblem::scalar_t c2 = math::cast<typename tproblem::scalar_t>(0.9),
                                typename tproblem::scalar_t beta = math::cast<typename tproblem::scalar_t>(0.7))
                        {
                                typedef typename tproblem::scalar_t     scalar_t;
                                typedef typename tproblem::vector_t     vector_t;

                                const scalar_t f0 = fx;
                                const scalar_t d0 = d.dot(gx);
                                const scalar_t step = c1 * d0;
                                const scalar_t curv = c2 * d0;
                                vector_t g;

                                // FIXME: this is not working well with GD, CGD, NR!!!

                                scalar_t t = math::cast<scalar_t>(1.0);
                                while ( problem.f(x + t * d, g) > f0 + t * step ||
                                        d.dot(g) < curv)
                                {
                                        t = beta * t;
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
                        const typename tproblem::vector_t& x0,
                        typename tproblem::history_t& history)
                {
                        typedef typename tproblem::scalar_t     scalar_t;
                        typedef typename tproblem::vector_t     vector_t;

                        if (problem.size() != math::cast<size_t>(x0.size()))
                        {
                                return false;
                        }

                        history.clear();

                        vector_t x = x0, gx, d;

                        // iterate until convergence
                        for (index_t i = 0; i < problem.iters(); i ++)
                        {
                                const scalar_t fx = problem.f(x, gx);
                                history.memo(x, fx, gx.norm());

                                // check convergence
                                if (impl::converged(gx, problem.eps()))
                                {
                                        break;
                                }

                                // descent direction
                                d = -gx;

                                // update solution
                                const scalar_t t = impl::line_search(problem, x, d, fx, gx);
                                x.noalias() += t * d;
                        }

                        return !history.empty();
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
                        const typename tproblem::vector_t& x0,
                        typename tproblem::history_t& history)
                {
                        typedef typename tproblem::scalar_t     scalar_t;
                        typedef typename tproblem::vector_t     vector_t;

                        if (problem.size() != math::cast<size_t>(x0.size()))
                        {
                                return false;
                        }

                        history.clear();

                        vector_t x = x0, gx, gx_prv, d, d_prv;

                        // iterate until convergence
                        for (index_t i = 0; i < problem.iters(); i ++)
                        {
                                const scalar_t fx = problem.f(x, gx);
                                history.memo(x, fx, gx.norm());

                                // check convergence
                                if (impl::converged(gx, problem.eps()))
                                {
                                        break;
                                }

                                // descent direction
                                if (i == 0)
                                {
                                        // initial: gradient descent
                                        d = -gx;
                                }
                                else
                                {
                                        // next: Polak–Ribière updates
                                        const scalar_t beta = gx.dot(gx - gx_prv) / (gx_prv.dot(gx_prv));
                                        d = -gx + std::max(static_cast<scalar_t>(0.0), beta) * d_prv;
                                }

                                // update solution
                                const scalar_t t = impl::line_search(problem, x, d, fx, gx);
                                x.noalias() += t * d;

                                gx_prv = gx;
                                d_prv = d;
                        }

                        return !history.empty();
                }

                /////////////////////////////////////////////////////////////////////////////////////////////
                // Newton-Raphson method starting from the initial value (guess) x0.
                /////////////////////////////////////////////////////////////////////////////////////////////

                template
                <
                        class tproblem
                >
                bool newton_raphson(
                        const tproblem& problem,
                        const typename tproblem::vector_t& x0,
                        typename tproblem::history_t& history)
                {
                        typedef typename tproblem::scalar_t     scalar_t;
                        typedef typename tproblem::vector_t     vector_t;
                        typedef typename tproblem::matrix_t     matrix_t;

                        if (problem.size() != math::cast<size_t>(x0.size()))
                        {
                                return false;
                        }

                        history.clear();

                        vector_t x = x0, gx, d;
                        matrix_t hx;

                        // iterate until convergence
                        for (index_t i = 0; i < problem.iters(); i ++)
                        {
                                const scalar_t fx = problem.f(x, gx, hx);
                                history.memo(x, fx, gx.norm());

                                // check convergence
                                if (impl::converged(gx, problem.eps()))
                                {
                                        break;
                                }

                                // descent direction
                                d = hx.fullPivLu().solve(-gx);

                                const scalar_t desc = -d.dot(gx);
                                if (desc < 2.0 * problem.eps())
                                {
                                        // switch to GD if the step direction is not a descent direction!
                                        d = -gx;
                                }

                                // update solution
                                const scalar_t t = impl::line_search(problem, x, d, fx, gx);
                                x.noalias() += t * d;
                        }

                        return !history.empty();
                }
        }
}

#endif // NANOCV_OPTIMIZE_H
