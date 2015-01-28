#pragma once

#include <vector>

namespace test
{
        ///
        /// \brief initialize matrix with random values
        ///
        template
        <
                typename tmatrix
        >
        void init_matrix(int rows, int cols, tmatrix& matrix)
        {
                matrix.resize(rows, cols);
                matrix.setRandom();
                matrix /= rows;
        }

        ///
        /// \brief initialize matrices with random values
        ///
        template
        <
                typename tmatrix
        >
        void init_matrices(int rows, int cols, int count, std::vector<tmatrix>& matrices)
        {
                matrices.resize(count);
                for (int i = 0; i < count; i ++)
                {
                        init_matrix(rows, cols, matrices[i]);
                }
        }

        ///
        /// \brief initialize matrices with zero
        ///
        template
        <
                typename tmatrix
        >
        void zero_matrices(std::vector<tmatrix>& matrices)
        {
                for (size_t i = 0; i < matrices.size(); i ++)
                {
                        matrices[i].setZero();
                }
        }

        ///
        /// \brief sum matrices
        ///
        template
        <
                typename tmatrix,
                typename tscalar = typename tmatrix::Scalar
        >
        tscalar sum_matrices(std::vector<tmatrix>& matrices)
        {
                tscalar sum = 0;
                for (size_t i = 0; i < matrices.size(); i ++)
                {
                        sum += matrices[i].sum();
                }
                return sum;
        }
}
