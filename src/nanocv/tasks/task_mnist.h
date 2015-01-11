#pragma once

#include "task.h"

namespace ncv
{
        ///
        /// MNIST task:
        ///      - digit classification
        ///      - 28x28 grayscale images as inputs
        ///      - 10 outputs (10 labels)
        ///
        /// http://yann.lecun.com/exdb/mnist/
        ///
        class mnist_task_t : public task_t
        {
        public:

                NANOCV_MAKE_CLONABLE(mnist_task_t, "MNIST (digit classification)")

                // constructor
                mnist_task_t(const string_t& configuration = string_t());

                // load images from the given directory
                virtual bool load(const string_t& dir) override;

                // access functions
                virtual size_t n_rows() const override { return 28; }
                virtual size_t n_cols() const override { return 28; }
                virtual size_t n_outputs() const override { return 10; }
                virtual size_t n_folds() const override { return 1; }
                virtual color_mode color() const override { return color_mode::luma; }

        private:

                // load binary file
                bool load(const string_t& ifile, const string_t& gfile, protocol p, size_t count);
        };
}
