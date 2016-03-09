#pragma once

#include "util.hpp"
#include "function.hpp"

namespace zob
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
                                return (x.array().square().square() - 16 * x.array().square() + 5 * x.array()).sum();
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                gx = 4 * x.array().cube() - 32 * x.array() + 5;

                                return fn_fval(x);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const tvector& x) const override
                {
                        return tscalar(-5.0) < x.minCoeff() && x.maxCoeff() < tscalar(5.0);
                }

                virtual bool is_minima(const tvector& x, const tscalar epsilon) const override
                {
                        const auto u1 = tscalar(-2.9035340);
                        const auto u2 = tscalar(+2.7468027);

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
