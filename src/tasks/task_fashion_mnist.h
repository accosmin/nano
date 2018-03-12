#pragma once

#include "task_mem_vision.h"

namespace nano
{
        ///
        /// fashion-MNIST task:
        ///      - fashion article classification
        ///      - 28x28 grayscale images as inputs
        ///      - 10 outputs (10 labels)
        ///
        /// https://github.com/zalandoresearch/fashion-mnist
        ///
        class fashion_mnist_task_t final : public mem_vision_task_t
        {
        public:

                fashion_mnist_task_t();
                bool populate() override;
                void to_json(json_t&) const final;
                void from_json(const json_t&) final;

        private:

                bool load_binary(const string_t& ifile, const string_t& gfile, const protocol, const size_t count);

                // attributes
                string_t        m_dir;  ///< directory where to load the task from
        };
}
