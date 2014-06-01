#ifndef NANOCV_TASK_MNIST_H
#define NANOCV_TASK_MNIST_H

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

                NANOCV_MAKE_CLONABLE(mnist_task_t)

                // constructor
                mnist_task_t(const string_t& = string_t())
                        :       task_t("MNIST (digit classification)")
                {
                }

                // load images from the given directory
                virtual bool load(const string_t& dir);

                // access functions
                virtual size_t n_rows() const { return 28; }
                virtual size_t n_cols() const { return 28; }
                virtual size_t n_outputs() const { return 10; }
                virtual size_t n_folds() const { return 1; }
                virtual color_mode color() const { return color_mode::luma; }

        private:

                // load binary file
                size_t load(const string_t& ifile, const string_t& gfile, protocol p);
        };
}

#endif // NANOCV_TASK_MNIST_H
