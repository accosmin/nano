#pragma once

#include "stringi.h"
#include "problem.h"

namespace nano
{
        ///
        /// \brief test optimization problem
        ///
        struct function_t
        {
                ///
                /// \brief destructor
                ///
                virtual ~function_t() {}

                ///
                /// \brief function name to identify it in tests and benchmarks
                ///
                virtual string_t name() const = 0;

                ///
                /// \brief construct the associated optimization problem
                ///
                virtual problem_t problem() const = 0;

                ///
                /// \brief check if a point is contained in the function's domain
                ///
                virtual bool is_valid(const vector_t& x) const = 0;

                ///
                /// \brief check if a point is epsilon-close to a known local minimum
                ///
                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const = 0;

                ///
                /// \brief check if function is convex
                ///
                virtual bool is_convex() const = 0;

                ///
                /// \brief range of valid dimensions
                ///
                virtual tensor_size_t min_dims() const = 0;
                virtual tensor_size_t max_dims() const = 0;
        };
}
