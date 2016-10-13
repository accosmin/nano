#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief create Bohachevsky test functions
        ///
        struct function_bohachevsky_t : public function_t
        {
                enum btype
                {
                        one,
                        two,
                        three
                };

                explicit function_bohachevsky_t(const btype type);

                virtual std::string name() const override final;
                virtual problem_t problem() const override final;
                virtual bool is_valid(const vector_t& x) const override final;
                virtual bool is_minima(const vector_t&, const scalar_t) const override final;
                virtual bool is_convex() const override final;
                virtual tensor_size_t min_dims() const override final;
                virtual tensor_size_t max_dims() const override final;

                btype   m_type;
        };
}
