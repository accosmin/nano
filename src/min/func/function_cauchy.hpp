#pragma once

#include "util.hpp"

namespace func
{
        ///
        /// \brief create Cauchy test functions
        ///
        template
        <
                typename tscalar_
        >
        struct function_cauchy_t
        {
                typedef min::problem_t<tscalar_>        tproblem;
                typedef typename tproblem::tsize        tsize;
                typedef typename tproblem::tscalar      tscalar;
                typedef typename tproblem::tvector      tvector;
                
                explicit function_cauchy_t(const tsize dims)
                        :       m_dims(dims)
                {
                }

                std::string name() const
                {
                        return "Cauchy" + std::to_string(m_dims) + "D";
                }

                tproblem problem() const
                {
                        const auto fn_size = [=] ()
                        {
                                return m_dims;
                        };

                        const auto fn_fval = [=] (const tvector& x)
                        {
                                return (1.0 + x.array().square()).log().sum();
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                gx = (2.0 * x.array()) / (1.0 + x.array().square());

                                return fn_fval(x);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                bool is_valid(const tvector&) const
                {
                        return true;
                }

                bool is_minima(const tvector& x, const tscalar epsilon) const
                {
                        return util::distance(x, tvector::Zero(m_dims)) < epsilon;
                }

                tsize   m_dims;
        };
}
