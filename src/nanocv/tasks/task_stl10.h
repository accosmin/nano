#pragma once

#include "task.h"

namespace ncv
{
        ///
        /// STL10 task:
        ///      - object classification
        ///      - 96x96 color images as inputs
        ///     - 10 outputs (10 labels)
        ///
        /// http://www.stanford.edu/~acoates/stl10/
        ///
        class stl10_task_t : public task_t
        {
        public:

                NANOCV_MAKE_CLONABLE(stl10_task_t, "STL-10 (object classification)")

                // constructor
                explicit stl10_task_t(const string_t& configuration = string_t());

                // load images from the given directory
                virtual bool load(const string_t& dir) override;

                // access functions
                virtual size_t n_rows() const override { return 96; }
                virtual size_t n_cols() const override { return 96; }
                virtual size_t n_outputs() const override { return 10; }
                virtual size_t n_folds() const override { return 10; }
                virtual color_mode color() const override { return color_mode::rgba; }
                                                   
        private:
                                                   
                // load binary files
                bool load_ifile(const string_t& filename, const char* bdata, size_t bdata_size, bool unlabed, size_t count);
                bool load_gfile(const string_t& filename, const char* bdata, size_t bdata_size, size_t count);

                // build folds
                bool load_folds(const string_t& filename, const char* bdata, size_t bdata_size, 
                                size_t n_test, size_t n_train, size_t n_unlabeled);
        };
}

