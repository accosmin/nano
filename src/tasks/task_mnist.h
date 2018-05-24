#pragma once

#include "task_mem_vision.h"

namespace nano
{
        enum class mnist_type
        {
                digits,
                ///
                /// MNIST task:
                ///      - digit classification
                ///      - 28x28 grayscale images as inputs
                ///      - 10 outputs (10 labels)
                ///
                /// http://yann.lecun.com/exdb/mnist/
                ///

                fashion
                ///
                /// fashion-MNIST task:
                ///      - fashion article classification
                ///      - 28x28 grayscale images as inputs
                ///      - 10 outputs (10 labels)
                ///
                /// https://github.com/zalandoresearch/fashion-mnist
                ///
        };

        ///
        /// \brief MNIST-like task.
        ///
        template <mnist_type ttype>
        class base_mnist_task_t final : public mem_vision_task_t
        {
        public:

                base_mnist_task_t();
                bool populate() override;
                void to_json(json_t&) const final;
                void from_json(const json_t&) final;

        private:

                bool load_binary(const string_t& ifile, const string_t& gfile, const protocol, const size_t count);

                // attributes
                string_t        m_dir;          ///< directory where to load the task from
                size_t          m_folds{10};    ///<
        };

        using mnist_task_t = base_mnist_task_t<mnist_type::digits>;
        using fashion_mnist_task_t = base_mnist_task_t<mnist_type::fashion>;
}
