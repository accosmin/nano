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

                const char* name() const
                {
                        switch (ttype)
                        {
                        case mnist_type::digits:        return "MNIST";
                        case mnist_type::fashion:       return "Fashion-MNIST";
                        default:                        return "???";
                        }
                }

                const char* dirname() const
                {
                        switch (ttype)
                        {
                        case mnist_type::digits:        return "/experiments/databases/mnist";
                        case mnist_type::fashion:       return "/experiments/databases/fashion-mnist";
                        default:                        return "";
                        }
                }

                strings_t labels() const
                {
                        switch (ttype)
                        {
                        case mnist_type::digits:        return
                                {
                                        "digit0", "digit1", "digit2", "digit3", "digit4",
                                        "digit5", "digit6", "digit7", "digit8", "digit9"
                                };
                        case mnist_type::fashion:       return
                                {
                                        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                                        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
                                };
                        default:                        return
                                {
                                        "???", "???", "???", "???", "???",
                                        "???", "???", "???", "???", "???"
                                };
                        }
                }

                bool load_binary(const string_t& ifile, const string_t& gfile, const protocol, const size_t count);

                // attributes
                string_t        m_dir;          ///< directory where to load the task from
                size_t          m_folds{10};    ///<
        };

        using mnist_task_t = base_mnist_task_t<mnist_type::digits>;
        using fashion_mnist_task_t = base_mnist_task_t<mnist_type::fashion>;
}
