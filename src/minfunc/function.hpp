#pragma once

#include "arch.h"
#include "min/problem.hpp"
#include <string>

namespace func
{            
        ///
        /// \brief test optimization problem
        ///
        template 
        <
                typename tscalar
        >                
        struct function_t
        {
                typedef min::problem_t<tscalar>         tproblem;
                typedef typename tproblem::tsize        tsize;
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

        ///
        /// \brief compute the infinity-norm of a vector
        ///
        template
        <
                typename tvector1
        >
        static decltype(auto) norm(const tvector1& a)
        {
                return a.template lpNorm<Eigen::Infinity>();
        }
        
        ///
        /// \brief compute the infinity-distance between two vectors
        ///
        template
        <
                typename tvector1,
                typename tvector2
        >
        static decltype(auto) distance(const tvector1& a, const tvector2& b)
        {
                return norm(a - b);
        }
}
