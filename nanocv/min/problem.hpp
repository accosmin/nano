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

                        /// disable for invalid types!
                        typename tvalid_tscalar = typename std::enable_if<std::is_floating_point<tscalar_>::value>::type
                >
                class problem_t
                {
                public:

                        typedef tscalar_                        tscalar;

                        /// current or the optimum optimization state
                        typedef state_t<tscalar>                tstate;

                        typedef typename tstate::tvector        tvector;
                        typedef typename tstate::tsize          tsize;

                        /// dimensionality operator: size = op()
                        typedef std::function<tsize()>                                  top_size;

                        /// function value operator: f = op(x)
                        typedef std::function<tscalar(const tvector&)>                  top_fval;

                        /// function value & gradient operator: f = op(x, g)
                        typedef std::function<tscalar(const tvector&, tvector&)>        top_grad;

                        /// logging operator: op(state), returns false if the optimization should stop
                        typedef std::function<bool(const tstate&)>                      top_ulog;

                        ///
                        /// \brief constructor (analytic gradient)
                        ///
                        template
                        <
                                typename topsize,
                                typename topfval,
                                typename topgrad
                        >
                        explicit problem_t(
                                const topsize& op_size,
                                const topfval& op_fval,
                                const topgrad& op_grad)
                                :       m_op_size(op_size),
                                        m_op_fval(op_fval),
                                        m_op_grad(op_grad)
                        {
                                clear();
                        }

                        ///
                        /// \brief constructor (no analytic gradient, can be estimated)
                        ///
                        template
                        <
                                typename topsize,
                                typename topfval
                        >
                        explicit problem_t(
                                const topsize& op_size,
                                const topfval& op_fval)
                                :       problem_t(op_size, op_fval, top_grad())
                        {
                        }

                        ///
                        /// \brief reset statistics (e.g. number of function value and gradient calls)
                        ///
                        void clear() const;

                        ///
                        /// \brief compute dimensionality
                        ///
                        tsize size() const;

                        ///
                        /// \brief compute function value
                        ///
                        tscalar operator()(const tvector& x) const;

                        ///
                        /// \brief compute function gradient
                        ///
                        tscalar operator()(const tvector& x, tvector& g) const;

                        ///
                        /// \brief number of function evalution calls
                        ///
                        tsize fcalls() const { return m_fcalls; }

                        ///
                        /// \brief number of function gradient calls
                        ///
                        tsize gcalls() const { return m_gcalls; }

                        ///
                        /// \brief compute the gradient accuracy (given vs. finite difference approximation)
                        ///
                        tscalar grad_accuracy(const tvector& x) const;

                private:

                        void eval_grad(const tvector& x, tvector& g) const;

                private:

                        // attributes
                        top_size                m_op_size;
                        top_fval                m_op_fval;
                        top_grad                m_op_grad;
                        mutable tsize           m_fcalls;               ///< #function value evaluations
                        mutable tsize           m_gcalls;               ///< #function gradient evaluations
                };

                template <typename ts, typename tv>
                void problem_t<ts, tv>::clear() const
                {
                        m_fcalls = 0;
                        m_gcalls = 0;
                }

                template <typename ts, typename tv>
                typename problem_t<ts, tv>::tsize problem_t<ts, tv>::size() const
                {
                        return m_op_size();
                }

                template <typename ts, typename tv>
                typename problem_t<ts, tv>::tscalar problem_t<ts, tv>::operator()(const tvector& x) const
                {
                        m_fcalls ++;
                        return m_op_fval(x);
                }

                template <typename ts, typename tv>
                typename problem_t<ts, tv>::tscalar problem_t<ts, tv>::operator()(const tvector& x, tvector& g) const
                {
                        if (m_op_grad)
                        {
                                m_fcalls ++;
                                m_gcalls ++;
                                return m_op_grad(x, g);
                        }
                        else
                        {
                                eval_grad(x, g);
                                return operator()(x);
                        }
                }

                template <typename ts, typename tv>
                typename problem_t<ts, tv>::tscalar problem_t<ts, tv>::grad_accuracy(const tvector& x) const
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

                template <typename ts, typename tv>
                void problem_t<ts, tv>::eval_grad(const tvector& x, tvector& g) const
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
        }
}

