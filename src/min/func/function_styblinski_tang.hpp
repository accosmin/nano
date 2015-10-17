#pragma once

#include "util.hpp"

namespace func
{
        ///
        /// \brief create Styblinski-Tang test functions
        ///
        template
        <       
                typename tscalar_
        >
        struct function_styblinski_tang_t
        {
                typedef min::problem_t<tscalar_>        tproblem;
                typedef typename tproblem::tsize        tsize;
                typedef typename tproblem::tscalar      tscalar;
                typedef typename tproblem::tvector      tvector;
                
                explicit function_styblinski_tang_t(const tsize dims)
                        :       m_dims(dims)
                {
                }

                std::string name() const
                {
                        return "Styblinski-Tang" + std::to_string(m_dims) + "D";
                }

                tproblem problem() const
                {
                        const auto fn_size = [=] ()
                        {
                                return m_dims;
                        };

                        const auto fn_fval = [=] (const tvector& x)
                        {
                                tscalar u = 0;
                                for (tsize i = 0; i < m_dims; i ++)
                                {
                                        u += util::quartic(x(i)) - 16 * util::square(x(i)) + 5 * x(i);
                                }

                                return 0.5 * u;
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                gx.resize(m_dims);
                                for (tsize i = 0; i < m_dims; i ++)
                                {
                                        gx(i) = 2 * util::cube(x(i)) - 16 * x(i) + 2.5;
                                }

                                return fn_fval(x);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                bool is_valid(const tvector& x) const
                {
                        return -5.0 < x.minCoeff() && x.maxCoeff() < 5.0;
                }

                bool is_minima(const tvector& x, const tscalar epsilon) const
                {
                        const tscalar u1 = -2.9035340;
                        const tscalar u2 = +2.7468027;

                        bool ok = true;
                        for (tsize i = 0; i < m_dims && ok; i ++)
                        {
                                ok = std::fabs(x(i) - u1) < epsilon || std::fabs(x(i) - u2) < epsilon;
                        }

                        return ok;
                }

                tsize   m_dims;
        };
}
