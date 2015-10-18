#pragma once

#include "min/problem.hpp"
#include <string>

namespace func
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
                typedef min::problem_t<tscalar_>        tproblem;
                typedef typename tproblem::tsize        tsize;
                typedef typename tproblem::tscalar      tscalar;
                typedef typename tproblem::tvector      tvector;
                
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
