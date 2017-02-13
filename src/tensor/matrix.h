#pragma once

#include <eigen3/Eigen/Core>
#include <type_traits>

namespace nano
{
        ///
        /// \brief matrix types
        ///
        template
        <
                typename tvalue_,
                int trows = Eigen::Dynamic,
                int tcols = Eigen::Dynamic,
                typename tvalue = typename std::remove_const<tvalue_>::type
        >
        using tensor_matrix_t = Eigen::Matrix<tvalue, trows, tcols, Eigen::RowMajor>;

        ///
        /// \brief map non-constant data to matrices
        ///
        template
        <
                int alignment = Eigen::Unaligned,
                typename tvalue_,
                typename tsize,
                typename tvalue = typename std::remove_const<tvalue_>::type,
                typename tresult = Eigen::Map<tensor_matrix_t<tvalue>, alignment>
        >
        tresult map_matrix(tvalue_* data, const tsize rows, const tsize cols)
        {
                return tresult(data, rows, cols);
        }

        ///
        /// \brief map constant data to matrices
        ///
        template
        <
                int alignment = Eigen::Unaligned,
                typename tvalue_,
                typename tsize,
                typename tvalue = typename std::remove_const<tvalue_>::type,
                typename tresult = Eigen::Map<const tensor_matrix_t<tvalue>, alignment>
        >
        tresult map_matrix(const tvalue_* data, const tsize rows, const tsize cols)
        {
                return tresult(data, rows, cols);
        }
}
