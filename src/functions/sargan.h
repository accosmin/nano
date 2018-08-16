#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief Sargan function: see http://infinity77.net/global_optimization/test_functions_nd_S.html.
        ///
        class function_sargan_t final : public function_t
        {
        public:

                explicit function_sargan_t(const tensor_size_t dims);

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
