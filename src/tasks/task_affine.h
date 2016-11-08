#pragma once

#include "task_mem_tensor.h"

namespace nano
{
        enum class affine_mode
        {
                regression,     ///< regression task: predict an affine transformation
                sign_class,     ///< multi-class classification task: classify the sign of an affine transformation
        };

        template <>
        inline std::map<affine_mode, string_t> enum_string<affine_mode>()
        {
                return
                {
                        { affine_mode::regression,      "regression" },
                        { affine_mode::sign_class,      "sign_class" }
                };
        }

        ///
        /// \brief task containing affine-mapped 3D input tensors: f(x) = A * x + b + noise.
        ///
        class NANO_PUBLIC affine_task_t final : public mem_tensor_task_t
        {
        public:

                explicit affine_task_t(const string_t& configuration = string_t());

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
