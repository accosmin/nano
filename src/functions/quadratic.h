#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief create quadratic test functions: f(x) = trans(a) * x + x * A * x
        ///
        class function_quadratic_t final : public function_t
        {
        public:
                explicit function_quadratic_t(const tensor_size_t dims);

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;

        private:

                // attributes
                vector_t        m_a;
                matrix_t        m_A;
        };
}
