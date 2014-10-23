#pragma once

#include "task.h"
#include "common/io_base.h"

namespace ncv
{
        ///
        /// CIFAR10 task:
        ///      - object classification
        ///      - 32x32 color images as inputs
        ///      - 10 outputs (10 labels)
        ///
        /// http://www.cs.toronto.edu/~kriz/cifar.html
        ///
        class cifar10_task_t : public task_t
        {
        public:

                NANOCV_MAKE_CLONABLE(cifar10_task_t, "CIFAR-10 (object classification)")

                // constructor
                cifar10_task_t(const string_t& configuration = string_t());

                // load images from the given directory
                virtual bool load(const string_t& dir);

                // access functions
                virtual size_t n_rows() const { return 32; }
                virtual size_t n_cols() const { return 32; }
                virtual size_t n_outputs() const { return 10; }
                virtual size_t n_folds() const { return 1; }
                virtual color_mode color() const { return color_mode::rgba; }

        private:

                // load binary file
                size_t load(const io::data_t& bfile, protocol p);
        };
}

