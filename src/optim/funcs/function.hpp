#pragma once

#include <string>
#include "optim/problem.h"

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
                virtual std::string name() const = 0;

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
        };
}
