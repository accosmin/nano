#pragma once

#include "util.hpp"
#include "function.hpp"

namespace nano
{
        ///
        /// \brief create rotated hyper-ellipsoid test functions
        ///
        struct function_rotated_ellipsoid_t : public function_t
        {
                explicit function_rotated_ellipsoid_t(const tensor_size_t dims) :
                        m_dims(dims),
                        m_weights(dims)
                {
                        for (tensor_size_t i = 0; i < dims; i ++)
                        {
                                m_weights(i) = static_cast<scalar_t>(dims - i);
                        }
                }

                virtual std::string name() const override
                {
                        return "rotated ellipsoid" + std::to_string(m_dims) + "D";
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
                        return util::norm(x) < 65.536;
                }

                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override
                {
                        return util::distance(x, vector_t::Zero(m_dims)) < epsilon;
                }

                tensor_size_t   m_dims;
                vector_t        m_weights;
        };
}
