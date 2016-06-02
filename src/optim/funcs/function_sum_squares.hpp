#pragma once

#include "util.hpp"
#include "function.hpp"

namespace nano
{
        ///
        /// \brief create sum of squares test functions
        ///
        struct function_sum_squares_t : public function_t
        {
                explicit function_sum_squares_t(const tsize dims) :
                        m_dims(dims),
                        m_weights(dims)
                {
                        for (tsize i = 0; i < dims; i ++)
                        {
                                m_weights(i) = static_cast<scalar_t>(i + 1);
                        }
                }

                virtual std::string name() const override
                {
                        return "sum squares" + std::to_string(m_dims) + "D";
                }

                virtual problem_t problem() const override
                {
                        const auto fn_size = [=] ()
                        {
                                return m_dims;
                        };

                        const auto fn_fval = [=] (const vector_t& x)
                        {
                                return (m_weights.array() * x.array().square()).sum();
                        };

                        const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                gx = 2 * m_weights.array() * x.array();

                                return fn_fval(x);
                        };

                        return {fn_size, fn_fval, fn_grad};
                }

                virtual bool is_valid(const vector_t& x) const override
                {
                        return util::norm(x) < scalar_t(5.12);
                }

                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override
                {
                        return util::distance(x, vector_t::Zero(m_dims)) < epsilon;
                }

                tsize   m_dims;
                vector_t m_weights;
        };
}
