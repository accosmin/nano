#pragma once

#include "util.hpp"
#include "function.hpp"

namespace math
{
        ///
        /// \brief create Zakharov test functions
        ///
        template
        <
                typename tscalar
        >
        struct function_zakharov_t : public function_t<tscalar>
        {
                using tsize = typename function_t<tscalar>::tsize;
                using tvector = typename function_t<tscalar>::tvector;
                using tproblem = typename function_t<tscalar>::tproblem;

                explicit function_zakharov_t(const tsize dims)
                        :       m_dims(dims),
                                m_weights(dims)
                {
                        for (tsize i = 0; i < dims; i ++)
                        {
                                m_weights(i) = tscalar(0.5) * i;
                        }
                }

                virtual std::string name() const override
                {
                        return "Zakharov" + std::to_string(m_dims) + "D";
                }

                virtual tproblem problem() const override
                {
                        const auto fn_size = [=] ()
                        {
                                return m_dims;
                        };

                        const auto fn_fval = [=] (const tvector& x)
                        {
                                const tscalar u = x.array().square().sum();
                                const tscalar v = (m_weights.array() * x.array()).sum();

                                return u + math::square(v) + math::quartic(v);
                        };

                        const auto fn_grad = [=] (const tvector& x, tvector& gx)
                        {
                                const tscalar u = x.array().square().sum();
                                const tscalar v = (m_weights.array() * x.array()).sum();

                                gx = 2 * x + (2 * v + 4 * math::cube(v)) * m_weights;

                                return u + math::square(v) + math::quartic(v);
                        };

                        return tproblem(fn_size, fn_fval, fn_grad);
                }

                virtual bool is_valid(const tvector& x) const override
                {
                        return tscalar(-5.0) < x.minCoeff() && x.maxCoeff() < tscalar(10.0);
                }

                virtual bool is_minima(const tvector& x, const tscalar epsilon) const override
                {
                        return util::distance(x, tvector::Zero(m_dims)) < epsilon;
                }

                tsize   m_dims;
                tvector m_weights;
        };  
}
