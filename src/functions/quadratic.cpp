#include "quadratic.h"

using namespace nano;

function_quadratic_t::function_quadratic_t(const tensor_size_t dims) :
        function_t("Quadratic", dims, 1, 100 * 1000, convexity::yes, 100),
        m_a(vector_t::Random(dims))
{
        // NB: generate random positive semi-definite matrix to keep the function convex
        matrix_t A = matrix_t::Random(dims, dims);
        m_A = matrix_t::Identity(dims, dims) + A * A.transpose();
}

scalar_t function_quadratic_t::vgrad(const vector_t& x, vector_t* gx) const
{
        if (gx)
        {
                *gx = m_a + m_A * x;
        }

        return x.transpose() * (m_a + (m_A * x) / scalar_t(2));
}
