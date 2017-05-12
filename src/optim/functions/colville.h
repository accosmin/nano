#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief create Colville test functions
        ///
        struct function_colville_t final : public function_t
        {
                function_colville_t();

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
