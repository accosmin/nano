#pragma once

#include "util.hpp"
#include "function.hpp"

namespace nano
{
        ///
        /// \brief create Styblinski-Tang test functions
        ///
        struct function_styblinski_tang_t : public function_t
        {
                explicit function_styblinski_tang_t(const tsize dims) :
                        m_dims(dims)
                {
                }

                virtual std::string name() const override
                {
                        return "Styblinski-Tang" + std::to_string(m_dims) + "D";
                }

                virtual problem_t problem() const override
                {
                        const auto fn_size = [=] ()
                        {
                                return m_dims;
                        };

                        const auto fn_fval = [=] (const vector_t& x)
                        {
                                return (x.array().square().square() - 16 * x.array().square() + 5 * x.array()).sum();
                        };

                        const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                gx = 4 * x.array().cube() - 32 * x.array() + 5;

                                return fn_fval(x);
                        };

                        return {fn_size, fn_fval, fn_grad};
                }

                virtual bool is_valid(const vector_t& x) const override
                {
                        return scalar_t(-5.0) < x.minCoeff() && x.maxCoeff() < scalar_t(5.0);
                }

                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override
                {
                        const auto u1 = scalar_t(-2.9035340);
                        const auto u2 = scalar_t(+2.7468027);

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
