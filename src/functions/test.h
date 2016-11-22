#pragma once

#include <memory>
#include "function.h"

namespace nano
{
        using rfunction_t = std::unique_ptr<function_t>;
        using rfunctions_t = std::vector<rfunction_t>;

        ///
        /// \brief construct test functions having the number of dimensions within the given range.
        ///
        NANO_PUBLIC rfunctions_t make_functions(const tensor_size_t min_dims, const tensor_size_t max_dims);

        ///
        /// \brief construct convex test functions having the number of dimensions within the given range.
        ///
        NANO_PUBLIC rfunctions_t make_convex_functions(const tensor_size_t min_dims, const tensor_size_t max_dims);

        ///
        /// \brief run the given operator for each test function.
        ///
        template <typename toperator>
        void foreach_test_function(const rfunctions_t& functions, const toperator& op)
        {
                for (const auto& func : functions)
                {
                        op(*func);
                }
        }
}
