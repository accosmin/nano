#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief Trid function: see https://www.sfu.ca/~ssurjano/trid.html.
        ///
        class function_trid_t final : public function_t
        {
        public:

                explicit function_trid_t(const tensor_size_t dims);

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
