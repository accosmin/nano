#include "sum_squares.h"

using namespace nano;

function_sum_squares_t::function_sum_squares_t(const tensor_size_t dims, const size_t N, const scalar_t lambda2) :
        function_t("SumSquares", dims, 1, 100 * 1000, convexity::no, 1),
        m_rng(make_rng()),
        m_udist(make_udist<size_t>(0, N - 1)),
        m_lambda2(lambda2),
        m_xis(N)
{
        for (auto& xi : m_xis)
        {
                xi = vector_t::Random(dims);
        }
}

scalar_t function_sum_squares_t::vgrad(const vector_t& x, vector_t* gx) const
{
        scalar_t fx = 0;
        for (const auto& xi : m_xis)
        {
                fx += (x - xi).dot(x - xi);
        }

        if (gx)
        {
                gx->setZero();
                for (const auto& xi : m_xis)
                {
                        *gx += x - xi;
                }
                *gx /= scalar_t(m_xis.size());
                *gx += m_lambda2 * x;
        }

        return fx / scalar_t(2 * m_xis.size()) + m_lambda2 * x.dot(x) / 2;
}

scalar_t function_sum_squares_t::stoch_vgrad(const vector_t& x, vector_t* gx, scalar_t& stoch_ratio) const
{
        const auto i = m_udist(m_rng);
        assert(i < m_xis.size());

        const auto& xi = m_xis[i];

        const auto fx = (x - xi).dot(x - xi) / 2;
        if (gx)
        {
                *gx = x - xi;
        }
        stoch_ratio = scalar_t(1) / scalar_t(m_xis.size());

        return fx;
}
