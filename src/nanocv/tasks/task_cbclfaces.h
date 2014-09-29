#pragma once

#include "task.h"

namespace ncv
{
        ///
        /// CBCL face task:
        ///      - face/non-face classification
        ///      - 19x19 grayscale images as inputs
        ///      - 2 outputs (binary classification)
        ///
        /// http://cbcl.mit.edu/software-datasets/FaceData2.html
        ///
        class cbclfaces_task_t : public task_t
        {
        public:

                NANOCV_MAKE_CLONABLE(cbclfaces_task_t)

                // constructor
                cbclfaces_task_t(const string_t& = string_t())
                        :       task_t("CBCL faces (face/non-face classification)")
                {
                }

                // load images from the given directory
                virtual bool load(const string_t& dir);

                // access functions
                virtual size_t n_rows() const { return 19; }
                virtual size_t n_cols() const { return 19; }
                virtual size_t n_outputs() const { return 1; }
                virtual size_t n_folds() const { return 1; }
                virtual color_mode color() const { return color_mode::luma; }

        private:

                // load files from the given directory
                size_t load(const string_t& dir, bool is_face, protocol p);
        };
}
