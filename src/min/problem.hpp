#pragma once

#include "state.hpp"
#include <functional>
#include <type_traits>

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

                using tscalar = tscalar_;

                /// current or the optimum optimization state
                using tstate = state_t<tscalar>;

                using tsize = typename tstate::tsize;
                using tvector = typename tstate::tvector;

                /// number of dimensions operator: size = op()
                using topsize = std::function<tsize()>;

                /// function value operator: f = op(x)
                using topfval = std::function<tscalar(const tvector&)>;

                /// function value & gradient operator: f = op(x, g)
                using topgrad = std::function<tscalar(const tvector&, tvector&)>;

                /// logging operator: op(state), returns false if the optimization should stop
                using topulog = std::function<bool(const tstate&)>;

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
                        const topsize& opsize,
                        const topfval& opfval,
                        const topgrad& opgrad)
                        :       m_opsize(opsize),
                                m_opfval(opfval),
                                m_opgrad(opgrad)
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
                        const topsize& opsize,
                        const topfval& opfval)
                        :       problem_t(opsize, opfval, topgrad())
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
                std::size_t fcalls() const { return m_fcalls; }

                ///
                /// \brief number of function gradient calls
                ///
                std::size_t gcalls() const { return m_gcalls; }

                ///
                /// \brief compute the gradient accuracy (given vs. finite difference approximation)
                ///
                tscalar grad_accuracy(const tvector& x) const;

        private:

                void eval_grad(const tvector& x, tvector& g) const;

        private:

                // attributes
                topsize                 m_opsize;
                topfval                 m_opfval;
                topgrad                 m_opgrad;
                mutable std::size_t     m_fcalls;               ///< #function value evaluations
                mutable std::size_t     m_gcalls;               ///< #function gradient evaluations
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
                return m_opsize();
        }

        template <typename ts, typename tv>
        typename problem_t<ts, tv>::tscalar problem_t<ts, tv>::operator()(const tvector& x) const
        {
                m_fcalls ++;
                return m_opfval(x);
        }

        template <typename ts, typename tv>
        typename problem_t<ts, tv>::tscalar problem_t<ts, tv>::operator()(const tvector& x, tvector& g) const
        {
                if (m_opgrad)
                {
                        m_fcalls ++;
                        m_gcalls ++;
                        return m_opgrad(x, g);
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
                if (m_opgrad)
                {
                        tvector gx;
                        const tscalar fx = m_opgrad(x, gx);

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

                        g(i) = (m_opfval(xp) - m_opfval(xn)) / (xp(i) - xn(i));
                }
        }
}

