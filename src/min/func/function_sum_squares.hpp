#pragma once

#include "util.hpp"

namespace func
{
        ///
        /// \brief create sum of squares test functions
        ///
        template 
        <
                typename tscalar_
        >
        struct function_sum_squares_t
        {
                typedef min::problem_t<tscalar_>        tproblem;
                typedef typename tproblem::tsize        tsize;
                typedef typename tproblem::tscalar      tscalar;
                typedef typename tproblem::tvector      tvector;
                
                explicit function_sum_squares_t(const tsize dims)
                        :       m_dims(dims)
                {
                }

                std::string name() const
                {
                        return "sum squares" + std::to_string(m_dims) + "D";
                }

                tproblem problem() const
                {
                        const auto fn_size = [=] ()
                        {
                                return m_dims;
                        };

                        const auto fn_fval = [=] (const tvector& x)
                        {
                                tscalar fx = 0;
                                for (tsize i = 0; i < m_dims; i ++)
                                {
                                        fx += (i + 1) * x(i) * x(i);
                                }

                                return fx;
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                gx.resize(m_dims);
                                for (tsize i = 0; i < m_dims; i ++)
                                {
                                        gx(i) = 2.0 * (i + 1) * x(i);
                                }

                                return fn_fval(x);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                bool is_valid(const tvector& x) const
                {
                        return util::norm(x) < 5.12;
                }

                bool is_minima(const tvector& x, const tscalar epsilon) const
                {
                        return util::distance(x, tvector::Zero(m_dims)) < epsilon;
                }

                tsize   m_dims;
        };
}
