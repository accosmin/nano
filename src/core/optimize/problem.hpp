#ifndef NANOCV_OPTIMIZE_PROBLEM_HPP
#define NANOCV_OPTIMIZE_PROBLEM_HPP

#include <functional>
#include "core/tensor/matrix.hpp"

namespace ncv
{
        namespace optimize
        {
                /////////////////////////////////////////////////////////////////////////////////////////////
                // describes a multivariate optimization problem.
                /////////////////////////////////////////////////////////////////////////////////////////////

                template
                <
                        typename tscalar,
                        typename tsize
                >
                class problem_t
                {
                public:

                        typedef typename tensor::vector_types_t<tscalar>::vector_t      vector_t;

                        // dimensionality operator
                        typedef std::function<tsize(void)>                              op_size_t;

                        // function value operator
                        typedef std::function<tscalar(const vector_t&)>                 op_fval_t;

                        // function value & gradient operator
                        typedef std::function<tscalar(const vector_t&, vector_t&)>      op_fval_grad_t;

                        // constructor (analytic gradient)
                        explicit problem_t(
                                const op_size_t& op_size,
                                const op_fval_t& op_fval,
                                const op_fval_grad_t& op_fval_grad)
                                :       m_op_size(op_size),
                                        m_op_fval(op_fval),
                                        m_op_fval_grad(op_fval_grad),
                                        m_n_fvals(0),
                                        m_n_grads(0)
                        {
                        }

                        // constructor (no analytic gradient, can be estimated)
                        explicit problem_t(
                                const op_size_t& op_size,
                                const op_fval_t& op_fval)
                                :       problem_t(op_size, op_fval, op_fval_grad_t())
                        {
                        }

                        // compute dimensionality & function value & gradient
                        tsize size() const { return _size(); }
                        tscalar f(const vector_t& x) const { return _f(x); }
                        tscalar f(const vector_t& x, vector_t& g) const { return _f(x, g); }

                        // access functions
                        tsize n_fval_calls() const { return m_n_fvals; }
                        tsize n_grad_calls() const { return m_n_grads; }

                private:

                        // implementation: dimensionality
                        tsize _size() const
                        {
                                return m_op_size();
                        }

                        // implementation: function value
                        tscalar _f(const vector_t& x) const
                        {
                                m_n_fvals ++;
                                return m_op_fval(x);
                        }

                        // implementation: function value & gradient
                        tscalar _f(const vector_t& x, vector_t& g) const
                        {
                                if (m_op_fval_grad)
                                {
                                        m_n_fvals ++;
                                        m_n_grads ++;
                                        return m_op_fval_grad(x, g);
                                }
                                else
                                {
                                        eval_grad(x, g);
                                        return f(x);
                                }
                        }

                        // implementation: approximate gradient (if no analytic gradient provided)
                        void eval_grad(const vector_t x, vector_t& g) const
                        {
                                const tsize n = size();
                                const tscalar d = 1e-6;//std::numeric_limits<tscalar>::epsilon();

                                vector_t xp = x, xn = x;

                                g.resize(n);
                                for (tsize i = 0; i < n; i ++)
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

                                g /= d * 2;
                        }

                private:

                        // attributes
                        op_size_t               m_op_size;
                        op_fval_t               m_op_fval;
                        op_fval_grad_t          m_op_fval_grad;
                        mutable tsize           m_n_fvals;              // #function value evaluations
                        mutable tsize           m_n_grads;              // #function gradient evaluations
                };
        }
}

#endif // NANOCV_OPTIMIZE_PROBLEM_HPP
