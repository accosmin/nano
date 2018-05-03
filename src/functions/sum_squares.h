#pragma once

#include "function.h"
#include "math/random.h"

namespace nano
{
        ///
        /// \brief f(x) = 1/N * sum(1/2 * ||x - x_i||^2, i=1,N) + lambda2/2 * ||x||^2,
        ///     with x_i randomly generated.
        ///
        class function_sum_squares_t final : public function_t
        {
        public:
                explicit function_sum_squares_t(const tensor_size_t dims, const size_t N, const scalar_t lambda2 = 0);

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
                scalar_t stoch_vgrad(const vector_t& x, vector_t* gx, scalar_t& stoch_ratio) const override;

        private:

                // attributes
                mutable rng_t           m_rng;          ///< RNG
                mutable udist_t<size_t> m_udist;        ///< [0, N) uniform distribution
                scalar_t                m_lambda2;      ///< L2-regularization factor
                std::vector<vector_t>   m_xis;          ///< {x_i}
        };
}
