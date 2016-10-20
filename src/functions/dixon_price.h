#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief create Dixon-Price test functions
        ///
        struct function_dixon_price_t final : public function_t
        {
                explicit function_dixon_price_t(const tensor_size_t dims);

                virtual std::string name() const override;
                virtual problem_t problem() const override;
                virtual bool is_valid(const vector_t& x) const override;
                virtual bool is_minima(const vector_t&, const scalar_t) const override;
                virtual bool is_convex() const override;
                virtual tensor_size_t min_dims() const override;
                virtual tensor_size_t max_dims() const override;

                tensor_size_t   m_dims;
        };
}
