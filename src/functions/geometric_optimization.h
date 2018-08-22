#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief generic geometric optimization function: f(x) = sum(i, alpha_i + a_i.dot(x)).
        ///
        ///     see "Introductory Lectures on Convex Optimization (Applied Optimization)",
        ///     by Y. Nesterov, 2013, p.56
        ///
        class function_geometric_optimization_t final : public function_t
        {
        public:

                explicit function_geometric_optimization_t(const tensor_size_t dims) :
                        function_t("Geometric Optimization", dims, 1, 100 * 1000, convexity::yes, 100),
                        m_a(vector_t::Random(dims)),
                        m_A(matrix_t::Random(dims, dims) / dims)
                {
                }

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override
                {
                        if (gx)
                        {
                                *gx = m_A.transpose() * (m_a + m_A * x).array().exp().matrix();
                        }

                        return (m_a + m_A * x).array().exp().sum();
                }

        private:

                // attributes
                vector_t        m_a;
                matrix_t        m_A;
        };
}
