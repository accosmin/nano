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

                virtual std::string name() const final;
                virtual problem_t problem() const final;
                virtual bool is_valid(const vector_t& x) const final;
                virtual bool is_minima(const vector_t&, const scalar_t) const final;
                virtual bool is_convex() const final;
                virtual tensor_size_t min_dims() const final;
                virtual tensor_size_t max_dims() const final;

                btype   m_type;
        };
}
