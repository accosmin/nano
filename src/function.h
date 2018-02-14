#pragma once

#include <regex>
#include <memory>
#include "arch.h"
#include "tensor.h"
#include "stringi.h"

namespace nano
{
        class function_t;
        using ref_function_t = std::reference_wrapper<const function_t>;
        using rfunction_t = std::unique_ptr<function_t>;
        using rfunctions_t = std::vector<rfunction_t>;

        enum class convexity
        {
                yes,
                no,
                unknown,
        };

        ///
        /// \brief construct test functions having the number of dimensions within the given range.
        ///
        NANO_PUBLIC rfunctions_t get_functions(
                const tensor_size_t min_dims, const tensor_size_t max_dims,
                const std::regex& name_regex = std::regex(".+"));

        ///
        /// \brief construct convex test functions having the number of dimensions within the given range.
        ///
        NANO_PUBLIC rfunctions_t get_convex_functions(
                const tensor_size_t min_dims, const tensor_size_t max_dims,
                const std::regex& name_regex = std::regex(".+"));

        ///
        /// \brief generic multi-dimensional optimization problem that
        ///     can be used with both batch and stochastic optimization algorithms:
        ///     - call function_t::vgrad() for the batch case
        ///     - call function_t::stoch_vgrad() and then function_t::stoch_next() for the stochastic case
        ///
        /// NB: by default the stochastic approximation is disabled (aka calling the batch version)
        ///
        class NANO_PUBLIC function_t
        {
        public:
                ///
                /// \brief constructor
                ///
                function_t(const char* name,
                        const tensor_size_t size, const tensor_size_t min_size, const tensor_size_t max_size,
                        const convexity convex,
                        const scalar_t domain);

                ///
                /// \brief destructor
                ///
                virtual ~function_t() = default;

                ///
                /// \brief function name to identify it in tests and benchmarks
                ///
                string_t name() const;

                ///
                /// \brief number of dimensions and the range of valid dimensions
                ///
                tensor_size_t size() const { return m_size; }
                tensor_size_t min_size() const { return m_min_size; }
                tensor_size_t max_size() const { return m_max_size; }

                ///
                /// \brief check if a point is within the function's domain
                ///
                bool is_valid(const vector_t& x) const;

                ///
                /// \brief check if function is convex
                ///
                bool is_convex() const { return m_convex == convexity::yes; }

                ///
                /// \brief check if the function is convex along the [x1, x2] line
                ///
                bool is_convex(const vector_t& x1, const vector_t& x2, const int steps) const;

                ///
                /// \brief compute function value (and gradient if provided)
                ///
                scalar_t eval(const vector_t& x, vector_t* gx = nullptr) const;

                ///
                /// \brief compute function value (and gradient if provided) using the stochastic approximation
                ///
                scalar_t stoch_eval(const vector_t& x, vector_t* gx = nullptr) const;

                ///
                /// \brief select another random minibatch
                ///
                virtual void stoch_next() const {}

                ///
                /// \brief number of function evaluation calls
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
                virtual scalar_t stoch_vgrad(const vector_t& x, vector_t* gx, scalar_t& stoch_ratio) const
                {
                        // no stochastic approximation by default
                        stoch_ratio = 1;
                        return vgrad(x, gx);
                }

        private:

                // attributes
                const char*     m_name;                         ///<
                tensor_size_t   m_size, m_min_size, m_max_size; ///< #dimensions
                convexity       m_convex;                       ///<
                scalar_t        m_domain;                       ///< domain = hyper-ball{0, m_domain}
                mutable scalar_t m_fcalls{0};                   ///< #function value evaluations
                mutable scalar_t m_gcalls{0};                   ///< #function gradient evaluations
        };
}
