#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief create Colville test functions
        ///
        struct NANO_PUBLIC function_colville_t : public function_t
        {
                virtual std::string name() const override;
                virtual problem_t problem() const override;
                virtual bool is_valid(const vector_t& x) const override;
                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override;
        };
}
