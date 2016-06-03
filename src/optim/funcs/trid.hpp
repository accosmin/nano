#pragma once

#include "util.hpp"
#include "function.hpp"

namespace nano
{
        ///
        /// \brief create Trid test functions
        ///
        struct function_trid_t : public function_t
        {
                explicit function_trid_t(const tensor_size_t dims) :
                        m_dims(dims)
                {
                }

                virtual std::string name() const override
                {
                        return "Trid" + std::to_string(m_dims) + "D";
                }

                virtual problem_t problem() const override
                {
                        const auto fn_size = [=] ()
                        {
                                return m_dims;
                        };

                        const auto fn_fval = [=] (const vector_t& x)
                        {
                                return (x.array() - 1).square().sum() -
                                       (x.segment(0, m_dims - 1).array() * x.segment(1, m_dims - 1).array()).sum();
                        };

                        const auto fn_grad = [=] (const vector_t& x, vector_t& gx)
                        {
                                gx = 2 * (x.array() - 1);
                                gx.segment(1, m_dims - 1) -= x.segment(0, m_dims - 1);
                                gx.segment(0, m_dims - 1) -= x.segment(1, m_dims - 1);

                                return fn_fval(x);
                        };

                        return {fn_size, fn_fval, fn_grad};
                }

                virtual bool is_valid(const vector_t& x) const override
                {
                        return util::norm(x) < scalar_t(1 + m_dims * m_dims);
                }

                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override
                {
                        vector_t xmin(m_dims);
                        for (tensor_size_t d = 0; d < m_dims; d ++)
                        {
                                xmin(d) = scalar_t(d + 1) * scalar_t(m_dims - d);
                        }

                        return util::distance(x, xmin) < epsilon;
                }

                tensor_size_t   m_dims;
        };
}
