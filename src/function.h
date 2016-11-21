#pragma once

#include "arch.h"
#include "tensor.h"
#include "stringi.h"

namespace nano
{
        class function_t;
        using ref_function_t = std::reference_wrapper<const function_t>;
        using rfunction_t = std::unique_ptr<function_t>;

        ///
        /// \brief generic multi-dimensional optimization problem that
        ///     can be used with both batch and stochastic optimization algorithms:
        ///     - call function_t::vgrad() for the batch case
        ///     - call function_t::stoch_vgrad() and then function_t::stoch_next() for the stochastic case
        ///
        class function_t
        {
        public:

                ///
                /// \brief default constructor
                ///
                function_t();

                ///
                /// \brief destructor
                ///
                virtual ~function_t() {}

                ///
                /// \brief function name to identify it in tests and benchmarks
                ///
                virtual string_t name() const = 0;

                ///
                /// \brief range of valid dimensions
                ///
                virtual tensor_size_t min_dims() const = 0;
                virtual tensor_size_t max_dims() const = 0;

                ///
                /// \brief number of dimensions
                ///
                virtual tensor_size_t dims() const = 0;

                ///
                /// \brief check if a point is contained in the function's domain
                ///
                virtual bool is_valid(const vector_t& x) const = 0;

                ///
                /// \brief check if a point is epsilon-close to a local minimum (if known)
                ///
                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const = 0;

                ///
                /// \brief check if function is convex
                ///
                virtual bool is_convex() const = 0;

                ///
                /// \brief check if the function is convex along the [x1, x2] line
                ///
                bool is_convex(const vector_t& x1, const vector_t& x2, const int steps) const;

                ///
                /// \brief compute function value (and gradient if provided)
                ///
                scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const;

                ///
                /// \brief compute function value (and gradient if provided) using the stochastic approximation
                ///
                scalar_t stoch_vgrad(const vector_t& x, vector_t* gx = nullptr) const;

                ///
                /// \brief select another random minibatch
                ///
                virtual void stoch_next() const = 0;

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

        protected:

                virtual scalar_t vgrad(const vector_t& x, vector_t* gx) const = 0;

                virtual size_t stoch_ratio() const = 0;
                virtual scalar_t stoch_vgrad(const vector_t& x, vector_t* gx) const = 0;

        private:

                void eval_grad(const vector_t& x, vector_t& g) const;

        private:

                // attributes
                mutable size_t          m_fcalls, m_stoch_fcalls;       ///< #function value evaluations
                mutable size_t          m_gcalls, m_stoch_gcalls;       ///< #function gradient evaluations
        };
}
