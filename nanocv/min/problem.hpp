#pragma once

#include "state.hpp"
#include <functional>
#include <type_traits>

namespace ncv
{
        namespace min
        {
                ///
                /// \brief describes a multivariate optimization problem
                ///
                template
                <
                        typename tscalar_,
                        typename tsize_,
                        typename top_size,              ///< dimensionality operator: size = op()
                        typename top_fval,              ///< function value operator: f = op(x)
                        typename top_grad,              ///< function value and gradient operator: f = op(x, g)

                        /// disable for not valid types!
                        typename tvalid_tscalar = typename std::enable_if<std::is_floating_point<tscalar_>::value>::type,
                        typename tvalid_tsize = typename std::enable_if<std::is_integral<tsize_>::value>::type
                >
                class problem_t
                {
                public:
                        typedef tscalar_                                tscalar;
                        typedef tsize_                                  tsize;

                        /// optimization current/optimum state
                        typedef state_t<tscalar, tsize>                 tstate;
                        typedef typename tstate::tvector                tvector;

                        /// log the current state
                        typedef std::function<bool(const tstate&)>      tulog;

                        ///
                        /// \brief constructor (analytic gradient)
                        ///
                        explicit problem_t(
                                const top_size& op_size,
                                const top_fval& op_fval,
                                const top_grad& op_grad)
                                :       m_op_size(op_size),
                                        m_op_fval(op_fval),
                                        m_op_grad(op_grad)
                        {
                                reset();
                        }

                        ///
                        /// \brief constructor (no analytic gradient, can be estimated)
                        ///
                        explicit problem_t(
                                const top_size& op_size,
                                const top_fval& op_fval)
                                :       problem_t(op_size, op_fval, top_grad())
                        {
                        }

                        ///
                        /// \brief reset statistics
                        ///
                        void reset() const
                        {
                                m_n_fvals = 0;
                                m_n_grads = 0;
                        }

                        ///
                        /// \brief compute dimensionality
                        ///
                        tsize size() const { return _size(); }

                        ///
                        /// \brief compute function value
                        ///
                        tscalar operator()(const tvector& x) const { return _f(x); }

                        ///
                        /// \brief compute function gradient
                        ///
                        tscalar operator()(const tvector& x, tvector& g) const { return _f(x, g); }

                        ///
                        /// \brief number of function evalution calls
                        ///
                        tsize n_fval_calls() const { return m_n_fvals; }

                        ///
                        /// \brief number of function gradient calls
                        ///
                        tsize n_grad_calls() const { return m_n_grads; }

                        ///
                        /// \brief compute the gradient accuracy (given vs. finite difference approximation)
                        ///
                        tscalar grad_accuracy(const tvector& x) const { return _grad_accuracy(x); }

                private:

                        // implementation: dimensionality
                        tsize _size() const
                        {
                                return m_op_size();
                        }

                        // implementation: function value
                        tscalar _f(const tvector& x) const
                        {
                                m_n_fvals ++;
                                return m_op_fval(x);
                        }

                        // implementation: function value & gradient
                        tscalar _f(const tvector& x, tvector& g) const
                        {
                                if (m_op_grad)
                                {
                                        m_n_fvals ++;
                                        m_n_grads ++;
                                        return m_op_grad(x, g);
                                }
                                else
                                {
                                        eval_grad(x, g);
                                        return _f(x);
                                }
                        }

                        // implementation: gradient accuracy
                        tscalar _grad_accuracy(const tvector& x) const
                        {
                                if (m_op_grad)
                                {
                                        tvector gx;
                                        const tscalar fx = m_op_grad(x, gx);

                                        tvector gx_approx;
                                        eval_grad(x, gx_approx);

                                        return  (gx - gx_approx).template lpNorm<Eigen::Infinity>() /
                                                (tscalar(1) + std::fabs(fx));
                                }
                                else
                                {
                                        return tscalar(0);
                                }
                        }

                        // implementation: approximate gradient (if no analytic gradient provided)
                        void eval_grad(const tvector& x, tvector& g) const
                        {
                                // accuracy epsilon as defined in:
                                //      see "Numerical optimization", Nocedal & Wright, 2nd edition, p.197
                                const tscalar dx = std::cbrt(tscalar(10) * std::numeric_limits<tscalar>::epsilon());

                                const tsize n = size();

                                tvector xp = x, xn = x;

                                g.resize(n);
                                for (tsize i = 0; i < n; i ++)
                                {
                                        if (i > 0)
                                        {
                                                xp(i - 1) -= dx;
                                                xn(i - 1) += dx;
                                        }
                                        xp(i) += dx;
                                        xn(i) -= dx;

                                        g(i) = (m_op_fval(xp) - m_op_fval(xn)) / (xp(i) - xn(i));
                                }
                        }

                private:

                        // attributes
                        top_size                m_op_size;
                        top_fval                m_op_fval;
                        top_grad                m_op_grad;
                        mutable tsize           m_n_fvals;              ///< #function value evaluations
                        mutable tsize           m_n_grads;              ///< #function gradient evaluations
                };
        }
}

