#pragma once

#include "tensor_mem.h"

namespace nano
{
        ///
        /// \brief map non-constant data to tensors
        ///
        template <typename tvalue_, std::size_t tdimensions>
        auto map_tensor(tvalue_* data, const tensor_dims_t<tdimensions>& dims)
        {
                using tvalue = typename std::remove_const<tvalue_>::type;
                return tensor_map_t<tvalue, tdimensions>(data, dims);
        }

        ///
        /// \brief map constant data to tensors
        ///
        template <typename tvalue_, std::size_t tdimensions>
        auto map_tensor(const tvalue_* data, const tensor_dims_t<tdimensions>& dims)
        {
                using tvalue = typename std::remove_const<tvalue_>::type;
                return tensor_const_map_t<tvalue, tdimensions>(data, dims);
        }

        ///
        /// \brief map non-constant data to tensors
        ///
        template <typename tvalue_, typename... tsizes>
        auto map_tensor(tvalue_* data, const tsizes... dims)
        {
                return map_tensor(data, make_dims(dims...));
        }

        ///
        /// \brief map constant data to tensors
        ///
        template <typename tvalue_, typename... tsizes>
        auto map_tensor(const tvalue_* data, const tsizes... dims)
        {
                return map_tensor(data, make_dims(dims...));
        }
}
