#pragma once

#include "util.hpp"
#include "function.hpp"

namespace nano
{
        ///
        /// \brief create Zakharov test functions
        ///
        struct function_zakharov_t : public function_t
        {
                explicit function_zakharov_t(const tsize dims) :
                        m_dims(dims),
                        m_weights(dims)
                {
                        for (tsize i = 0; i < dims; i ++)
                        {
                                m_weights(i) = static_cast<scalar_t>(i / 2);
                        }
                }

                virtual std::string name() const override
                {
                        return "Zakharov" + std::to_string(m_dims) + "D";
                }

                virtual problem_t problem() const override
                {
                        const auto fn_size = [=] ()
                        {
                                return m_dims;
                        };

                        const auto fn_fval = [=] (const vector_t& x)
                        {
                                const scalar_t u = x.array().square().sum();
                                const scalar_t v = (m_weights.array() * x.array()).sum();

                                return u + nano::square(v) + nano::quartic(v);
                        };

                        const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                const scalar_t u = x.array().square().sum();
                                const scalar_t v = (m_weights.array() * x.array()).sum();

                                gx = 2 * x + (2 * v + 4 * nano::cube(v)) * m_weights;

                                return u + nano::square(v) + nano::quartic(v);
                        };

                        return {fn_size, fn_fval, fn_grad};
                }

                virtual bool is_valid(const vector_t& x) const override
                {
                        return scalar_t(-5.0) < x.minCoeff() && x.maxCoeff() < scalar_t(10.0);
                }

                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override
                {
                        return util::distance(x, vector_t::Zero(m_dims)) < epsilon;
                }

                tsize           m_dims;
                vector_t        m_weights;
        };
}
