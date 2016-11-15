#pragma once

#include "arch.h"
#include "tensor.h"
#include <functional>

namespace nano
{
        ///
        /// \brief describes a multivariate optimization problem.
        ///     - the function value and gradient are computed using the provided operators
        ///     - stochastic approximation operators can be provided (useful for stochastic optimization),
        ///             otherwise the batch operators will be instead
        ///
        class NANO_PUBLIC problem_t
        {
        public:

                /// number of dimensions operator: size = op()
                using opsize_t = std::function<vector_t::Index()>;

                /// function value operator: f = op(x)
                using opfval_t = std::function<scalar_t(const vector_t&)>;

                /// function value & gradient operator: f = op(x, g)
                using opgrad_t = std::function<scalar_t(const vector_t, vector_t&)>;

                ///
                /// \brief constructor (analytic gradient)
                ///
                problem_t(
                        const opsize_t& opsize,
                        const opfval_t& opfval,
                        const opgrad_t& opgrad);

                ///
                /// \brief constructor (no analytic gradient, but can be inefficiently estimated)
                ///
                problem_t(
                        const opsize_t& opsize,
                        const opfval_t& opfval);

                ///
                /// \brief constructor (analytic gradient with stochastic approximations)
                ///
                problem_t(
                        const opsize_t& opsize,
                        const opfval_t& opfval,
                        const opgrad_t& opgrad,
                        const opfval_t& stoch_opfval,
                        const opgrad_t& stoch_opgrad,
                        const size_t stoch_ratio);

                ///
                /// \brief reset statistics (e.g. number of function value and gradient calls)
                ///
                void clear() const;

                ///
                /// \brief number of dimensions
                ///
                tensor_size_t size() const;

                ///
                /// \brief compute function value
                ///
                scalar_t value(const vector_t& x) const;

                ///
                /// \brief compute function value (using the stochastic approximation, if provided)
                ///
                scalar_t stoch_value(const vector_t& x) const;

                ///
                /// \brief compute function value and gradient
                ///
                scalar_t vgrad(const vector_t& x, vector_t& g) const;

                ///
                /// \brief compute function value and gradient (using the stochastic approximation, if provided)
                ///
                scalar_t stoch_vgrad(const vector_t& x, vector_t& g) const;

                ///
                /// \brief number of function evalution calls
                ///
                size_t fcalls() const;

                ///
                /// \brief number of function gradient calls
                ///
                size_t gcalls() const;

                ///
                /// \brief compute the gradient accuracy (given vs. finite difference approximation)
                ///
                scalar_t grad_accuracy(const vector_t& x) const;

                ///
                /// \brief check if the function is convex along the [x1, x2] line
                ///
                bool is_convex(const vector_t& x1, const vector_t& x2, const int steps) const;

        private:

                void eval_grad(const vector_t& x, vector_t& g) const;

        private:

                // attributes
                opsize_t                m_opsize;
                opfval_t                m_opfval;
                opgrad_t                m_opgrad;
                size_t                  m_stoch_ratio;                  ///< #stochastic calls per batch call
                opfval_t                m_stoch_opfval;                 ///< stochastic approx of function value
                opgrad_t                m_stoch_opgrad;                 ///< stochastic approx of function gradient
                mutable size_t          m_fcalls, m_stoch_fcalls;       ///< #function value evaluations
                mutable size_t          m_gcalls, m_stoch_gcalls;       ///< #function gradient evaluations
        };
}

