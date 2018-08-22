#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief generic lp-norm approximation: f(x) = sum(i, (a_i.dot(x) - b_i)^p), where p > 1.
        ///
        ///     see "Introductory Lectures on Convex Optimization (Applied Optimization)",
        ///     by Y. Nesterov, 2013, p.56
        ///
        class function_lpnorm_approximation_t final : public function_t
        {
        public:

                explicit function_lpnorm_approximation_t(const tensor_size_t dims) :
                        function_t("Lp-Norm Approximation", dims, 1, 100 * 1000, convexity::yes, 100),
                        m_b(vector_t::Random(dims)),
                        m_A(matrix_t::Random(dims, dims) / dims)
                {
                }

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override
                {
                        const auto p = scalar_t(1.5);

                        if (gx)
                        {
                                const auto diff = (m_A * x - m_b).array();
                                *gx = p * m_A.transpose() * (diff.abs().pow(p - 1) * diff.sign()).matrix();
                        }

                        return (m_A * x - m_b).array().abs().pow(p).sum();
                }

        private:

                // attributes
                vector_t        m_b;
                matrix_t        m_A;
        };
}
