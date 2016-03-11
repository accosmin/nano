#pragma once

#include "math/problem.hpp"
#include <string>

namespace nano
{            
        ///
        /// \brief test optimization problem
        ///
        template 
        <
                typename tscalar_
        >
        struct function_t
        {
                using tproblem = nano::problem_t<tscalar_>;
                using tsize = typename tproblem::tsize;
                using tscalar = typename tproblem::tscalar;
                using tvector = typename tproblem::tvector;

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
                virtual tproblem problem() const = 0;

                ///
                /// \brief check if a point is contained in the function's domain
                ///
                virtual bool is_valid(const tvector& x) const = 0;

                ///
                /// \brief check if a point is epsilon-close to a known local minimum
                ///
                virtual bool is_minima(const tvector& x, const tscalar epsilon) const = 0;
        };
}
