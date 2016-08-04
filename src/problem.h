#pragma once

#include "arch.h"
#include "tensor.h"
#include <functional>

namespace nano
{
        ///
        /// \brief describes a multivariate optimization problem
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
                /// \brief constructor (no analytic gradient, can be estimated)
                ///
                problem_t(
                        const opsize_t& opsize,
                        const opfval_t& opfval);

                ///
                /// \brief reset statistics (e.g. number of function value and gradient calls)
                ///
                void clear() const;

                ///
                /// \brief compute dimensionality
                ///
                tensor_size_t size() const;

                ///
                /// \brief compute function value
                ///
                scalar_t operator()(const vector_t& x) const;

                ///
                /// \brief compute function gradient
                ///
                scalar_t operator()(const vector_t& x, vector_t& g) const;

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
                mutable std::size_t     m_fcalls;               ///< #function value evaluations
                mutable std::size_t     m_gcalls;               ///< #function gradient evaluations
        };
}

