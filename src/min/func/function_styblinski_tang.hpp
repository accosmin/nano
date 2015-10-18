#pragma once

#include "function.hpp"

namespace min
{
        ///
        /// \brief create Styblinski-Tang test functions
        ///
        template
        <       
                typename tscalar
        >
        struct function_styblinski_tang_t : public function_t<tscalar>
        {
                using tsize = typename function_t<tscalar>::tsize;
                using tvector = typename function_t<tscalar>::tvector;
                using tproblem = typename function_t<tscalar>::tproblem;
                
                explicit function_styblinski_tang_t(const tsize dims)
                        :       m_dims(dims)
                {
                }

                virtual std::string name() const override
                {
                        return "Styblinski-Tang" + std::to_string(m_dims) + "D";
                }

                virtual tproblem problem() const override
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

                virtual bool is_valid(const tvector& x) const override
                {
                        return -5.0 < x.minCoeff() && x.maxCoeff() < 5.0;
                }

                virtual bool is_minima(const tvector& x, const tscalar epsilon) const override
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
