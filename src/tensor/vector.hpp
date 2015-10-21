#pragma once

#include <type_traits>
#include <eigen3/Eigen/Core>

namespace tensor
{
        ///
        /// \brief vector types
        ///
        template
        <
                typename tvalue_,
                typename tvalue = typename std::remove_const<tvalue_>::type
        >
        using vector_t = Eigen::Matrix<tvalue, Eigen::Dynamic, 1, Eigen::ColMajor>;

        ///
        /// \brief fixed size vector types
        ///
        template
        <
                typename tvalue_,
                int trows,
                typename tvalue = typename std::remove_const<tvalue_>::type
        >
        using fixed_size_vector_t = Eigen::Matrix<tvalue, trows, 1, Eigen::ColMajor>;

        ///
        /// \brief map non-constant data to vectors
        ///
        template
        <
                int alignment = Eigen::Unaligned,
                typename tvalue_,
                typename tsize,
                typename tvalue = typename std::remove_const<tvalue_>::type,
                typename tresult = Eigen::Map<vector_t<tvalue>, alignment>
        >
        tresult map_vector(tvalue_* data, tsize rows)
        {
                return tresult(data, rows);
        }

        ///
        /// \brief map constant data to vectors
        ///
        template
        <
                int alignment = Eigen::Unaligned,
                typename tvalue_,
                typename tsize,
                typename tvalue = typename std::remove_const<tvalue_>::type,
                typename tresult = Eigen::Map<const vector_t<tvalue>, alignment>
        >
        tresult map_vector(const tvalue_* data, tsize rows)
        {
                return tresult(data, rows);
        }
}

