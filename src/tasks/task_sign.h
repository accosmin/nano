#pragma once

#include "task_mem_tensor.h"

namespace nano
{
        ///
        /// \brief task to classify based on the sign the affine-mapped inputs: f(x) = A * x + b + noise.
        ///
        class NANO_PUBLIC sign_task_t final : public mem_tensor_task_t
        {
        public:

                explicit sign_task_t(const string_t& configuration = string_t());

                ///
                /// \brief return the affine parameters used for generating samples
                ///
                const matrix_t& weights() const { return m_A; }
                const vector_t& bias() const { return m_b; }

        private:

                virtual bool populate() override;

        private:

                matrix_t        m_A;
                vector_t        m_b;
        };
}
