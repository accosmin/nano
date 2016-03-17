#pragma once

#include "task_mem_vision.h"

namespace nano
{
        ///
        /// STL10 task:
        ///      - object classification
        ///      - 96x96 color images as inputs
        ///     - 10 outputs (10 labels)
        ///
        /// http://www.stanford.edu/~acoates/stl10/
        ///
        class stl10_task_t : public mem_vision_task_t
        {
        public:

                NANO_MAKE_CLONABLE(stl10_task_t, "STL-10 (object classification)")

                ///
                /// \brief constructor
                ///
                explicit stl10_task_t(const string_t& configuration = string_t());

                ///
                /// \brief short name of this task
                ///
                virtual string_t name() const override { return "stl-10"; }

                ///
                /// \brief load the task from the given directory (if possible)
                ///
                virtual bool load(const string_t& dir = string_t()) override;

        private:

                // load binary files
                bool load_ifile(const string_t& filename, const char* bdata, size_t bdata_size, bool unlabed, size_t count);
                bool load_gfile(const string_t& filename, const char* bdata, size_t bdata_size, size_t count);

                // build folds
                bool load_folds(const string_t& filename, const char* bdata, size_t bdata_size,
                                size_t n_test, size_t n_train, size_t n_unlabeled);
        };
}

