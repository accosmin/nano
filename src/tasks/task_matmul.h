#pragma once

#include "task_mem_tensor.h"

namespace nano
{
        ///
        /// \brief task to predict the result of a matrix multiplication: f(X1, X2) = A * X1 * X2  + B + noise.
        /// NB: the input tensors are of size [2, rows, cols] where each plane contains one of the matrices to multiply.
        ///
        class NANO_PUBLIC matmul_task_t final : public mem_tensor_task_t
        {
        public:

                explicit matmul_task_t(const string_t& configuration = string_t());

                ///
                /// \brief return the matrices used for generating samples
                ///
                const matrix_t& weights() const { return m_A; }
                const matrix_t& bias() const { return m_B; }

        private:

                virtual bool populate() override;

        private:

                matrix_t        m_A;
                matrix_t        m_B;
        };
}
