#pragma once

#include "arch.h"
#include "core/string.h"
#include "core/optimizer.h"
#include <memory>

namespace ncv
{
        struct function_t;
        typedef std::shared_ptr<function_t>             rfunction_t;
        typedef std::vector<rfunction_t>                functions_t;                

        ///
        /// \brief test optimization problem
        ///
        struct NANOCV_PUBLIC function_t
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
                virtual opt_problem_t problem() const = 0;

                ///
                /// \brief check if a point is contained in the function's domain
                ///
                virtual bool is_valid(const opt_vector_t& x) const = 0;

                ///
                /// \brief check if a point is epsilon-close to a known local minima
                ///
                virtual bool is_minima(const opt_vector_t& x, const opt_scalar_t epsilon) const = 0;

                ///
                /// \brief compute the infinity-distance between two vectors
                ///
                template
                <
                        typename tvector1,
                        typename tvector2
                >
                static opt_scalar_t distance(const tvector1& a, const tvector2& b)
                {
                        return norm(a - b);
                }

                ///
                /// \brief compute the infinity-norm of a vector
                ///
                template
                <
                        typename tvector
                >
                static opt_scalar_t norm(const tvector& a)
                {
                        return a.template lpNorm<Eigen::Infinity>();
                }
        };
}
