#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief create Booth test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        struct function_booth_t : public function_t
        {
                virtual std::string name() const override;
                virtual problem_t problem() const override;
                virtual bool is_valid(const vector_t& x) const override;
                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override;
                virtual bool is_convex() const override;
                virtual tensor_size_t min_dims() const override;
                virtual tensor_size_t max_dims() const override;
        };
}
