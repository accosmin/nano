#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief create Goldstein-Price test functions
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        struct NANO_PUBLIC function_goldstein_price_t : public function_t
        {
                virtual std::string name() const override;
                virtual problem_t problem() const override;
                virtual bool is_valid(const vector_t& x) const override;
                virtual bool is_minima(const vector_t& x, const scalar_t epsilon) const override;
        };
}
