#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief Chung-Reynolds function: f(x) = (x.dot(x))^2.
        ///
        class function_chung_reynolds_t final : public function_t
        {
        public:

                explicit function_chung_reynolds_t(const tensor_size_t dims);

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
