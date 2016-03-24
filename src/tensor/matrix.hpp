#pragma once

#include <eigen3/Eigen/Core>
#include <type_traits>

namespace tensor
{
        ///
        /// \brief matrix types
        ///
        template
        <
                typename tvalue_,
                typename tvalue = typename std::remove_const<tvalue_>::type
        >
        using matrix_t = Eigen::Matrix<tvalue, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

        ///
        /// \brief fixed size matrix types
        ///
        template
        <
                typename tvalue_,
                int trows,
                int tcols,
                typename tvalue = typename std::remove_const<tvalue_>::type
        >
        using fixed_size_matrix_t = Eigen::Matrix<tvalue, trows, tcols, Eigen::RowMajor>;

        ///
        /// \brief map non-constant data to matrices
        ///
        template
        <
                int alignment = Eigen::Unaligned,
                typename tvalue_,
                typename tsize,
                typename tvalue = typename std::remove_const<tvalue_>::type,
                typename tresult = Eigen::Map<matrix_t<tvalue>, alignment>
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
                typename tresult = Eigen::Map<const matrix_t<tvalue>, alignment>
        >
        tresult map_matrix(const tvalue_* data, const tsize rows, const tsize cols)
        {
                return tresult(data, rows, cols);
        }
}
