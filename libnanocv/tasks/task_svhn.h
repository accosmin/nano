#pragma once

#include "libnanocv/task.h"
#include "libnanocv/file/base.h"

namespace ncv
{
        ///
        /// SVHN task:
        ///      - digit classification
        ///      - 32x32 color images as inputs
        ///      - 10 outputs (10 labels)
        ///
        /// http://ufldl.stanford.edu/housenumbers/
        ///
        class svhn_task_t : public task_t
        {
        public:

                NANOCV_MAKE_CLONABLE(svhn_task_t, "SVHN (object classification)")

                // constructor
                explicit svhn_task_t(const string_t& configuration = string_t());

                // load images from the given directory
                virtual bool load(const string_t& dir) override;

                // access functions
                virtual size_t n_rows() const override { return 32; }
                virtual size_t n_cols() const override { return 32; }
                virtual size_t n_outputs() const override { return 10; }
                virtual size_t n_folds() const override { return 1; }
                virtual color_mode color() const override { return color_mode::rgba; }

        private:

                // load binary file
                size_t load(const string_t& bfile, protocol p);

                // decode the uncompressed bytes (images + labels)
                size_t decode(const io::data_t& image_data, const io::data_t& label_data, protocol p);
        };
}

