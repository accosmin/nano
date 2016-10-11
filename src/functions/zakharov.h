#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief create Zakharov test functions
        ///
        struct function_zakharov_t : public function_t
        {
                explicit function_zakharov_t(const tensor_size_t dims);

                virtual std::string name() const final;
                virtual problem_t problem() const final;
                virtual bool is_valid(const vector_t& x) const final;
                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const final;
                virtual bool is_convex() const final;
                virtual tensor_size_t min_dims() const final;
                virtual tensor_size_t max_dims() const final;

                tensor_size_t   m_dims;
        };
}
