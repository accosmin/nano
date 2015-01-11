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

                NANOCV_MAKE_CLONABLE(cbclfaces_task_t, "CBCL faces (face/non-face classification)")

                // constructor
                cbclfaces_task_t(const string_t& configuration = string_t());

                // load images from the given directory
                virtual bool load(const string_t& dir) override;

                // access functions
                virtual size_t n_rows() const override { return 19; }
                virtual size_t n_cols() const override { return 19; }
                virtual size_t n_outputs() const override { return 1; }
                virtual size_t n_folds() const override { return 1; }
                virtual color_mode color() const override { return color_mode::luma; }
        };
}
