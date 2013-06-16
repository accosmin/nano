#ifndef NANOCV_OPTIMIZE_H
#define NANOCV_OPTIMIZE_H

#include "stats.h"

namespace ncv
{
        namespace optimize
        {
                /////////////////////////////////////////////////////////////////////////////////////////////
                // optimization state: current point (x), function value (f),
                //      gradient (g) and descent direction (d).
                ////////////////////////////////////////////////////////////////////////////////////////////////

                struct state_t
                {
                        // constructor
                        state_t(size_t size = 0)
                                : x(size),
                                  g(size),
                                  d(size),
                                  f(std::numeric_limits<scalar_t>::max())
                        {
                        }

                        // constructor
                        template
                        <
                                class problem_t
                        >
                        state_t(const problem_t& problem, const vector_t& x0)
                        {
                                x = x0;
                                f = problem.f(x, g);
                        }

                        // update current point
                        template
                        <
                                class problem_t
                        >
                        void update(const problem_t& problem, scalar_t t)
                        {
                                x.noalias() += t * d;
                                f = problem.f(x, g);
                        }

                        // attributes
                        vector_t x, g, d;
                        scalar_t f;
                };

                // compare two optimization states
                inline bool operator<(const state_t& one, const state_t& other)
                {
                        return one.f < other.f;
                }

                /////////////////////////////////////////////////////////////////////////////////////////////
                // optimization result:
                //      solution & statistics
                /////////////////////////////////////////////////////////////////////////////////////////////

                class result_t
                {
                public:

                        // constructor (analytic gradient)
                        explicit result_t(size_t size);

                        // update solution
                        void update(const state_t& st);

                        // access functions
                        const state_t& optimum() const { return m_optimum; }
                        const stats_t& speed() const { return m_conv_speed; }
                        size_t iterations() const { return m_iterations; }

                private:

                        // attributes
                        state_t         m_optimum;              // optimum state
                        size_t          m_iterations;           // #iterations
                        stats_t         m_conv_speed;           // convergence speed statistics
                };

                /////////////////////////////////////////////////////////////////////////////////////////////
                // describes a multivariate optimization problem.
                /////////////////////////////////////////////////////////////////////////////////////////////

                // dimensionality
                typedef std::function<size_t(void)>                                     op_size_t;

                // function value
                typedef std::function<scalar_t(const vector_t&)>                        op_fval_t;

                // function value & gradient
                typedef std::function<scalar_t(const vector_t&, vector_t&)>             op_fval_grad_t;

                class problem_t
                {
                public:

                        // constructor (analytic gradient)
                        explicit problem_t(
                                const op_size_t& op_size,
                                const op_fval_t& op_fval,
                                const op_fval_grad_t& op_fval_grad);

                        // constructor (no analytic gradient, can be estimated)
                        explicit problem_t(
                                const op_size_t& op_size,
                                const op_fval_t& op_fval);

                        // compute dimensionality & function value & gradient
                        size_t size() const;
                        scalar_t f(const vector_t& x) const;
                        scalar_t f(const vector_t x, vector_t& g) const;

                private:

                        // evaluate gradient (if not provided)
                        void eval_grad(const vector_t x, vector_t& g) const;

                private:

                        // attributes
                        op_size_t               m_op_size;
                        op_fval_t               m_op_fval;
                        op_fval_grad_t          m_op_fval_grad;
                };

                // result updated
                typedef std::function<void(const result_t&)>    op_updated_t;

                /////////////////////////////////////////////////////////////////////////////////////////////
                // gradient descent starting from the initial value (guess) x0.
                /////////////////////////////////////////////////////////////////////////////////////////////

                result_t gradient_descent(
                        const problem_t& problem,
                        const vector_t& x0,
                        size_t max_iterations,          // maximum number of iterations
                        scalar_t epsilon,               // convergence precision
                        const op_updated_t& op_updated = op_updated_t());

                /////////////////////////////////////////////////////////////////////////////////////////////
                // conjugate gradient descent starting from the initial value (guess) x0.
                /////////////////////////////////////////////////////////////////////////////////////////////

                result_t conjugate_gradient_descent(
                        const problem_t& problem,
                        const vector_t& x0,
                        size_t max_iterations,          // maximum number of iterations
                        scalar_t epsilon,               // convergence precision
                        const op_updated_t& op_updated = op_updated_t());

                /////////////////////////////////////////////////////////////////////////////////////////////
                // limited memory bfgs (l-bfgs) starting from the initial value (guess) x0.
                /////////////////////////////////////////////////////////////////////////////////////////////

                result_t lbfgs(
                        const problem_t& problem,
                        const vector_t& x0,
                        size_t max_iterations,          // maximum number of iterations
                        scalar_t epsilon,               // convergence precision
                        size_t history_size = 8,        // hessian approximation history size
                        const op_updated_t& op_updated = op_updated_t());
        }
}

#endif // NANOCV_OPTIMIZE_H
