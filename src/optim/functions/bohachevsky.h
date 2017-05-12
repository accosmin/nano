#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief create Bohachevsky test functions.
        ///
        struct function_bohachevsky1_t final : public function_t
        {
                explicit function_bohachevsky1_t();

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };

        struct function_bohachevsky2_t final : public function_t
        {
                explicit function_bohachevsky2_t();

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };

        struct function_bohachevsky3_t final : public function_t
        {
                explicit function_bohachevsky3_t();

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
