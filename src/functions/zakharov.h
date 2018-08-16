#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief Zakharov function: see https://www.sfu.ca/~ssurjano/zakharov.html.
        ///
        class function_zakharov_t final : public function_t
        {
        public:

                explicit function_zakharov_t(const tensor_size_t dims);

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
