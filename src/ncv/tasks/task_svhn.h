#ifndef NANOCV_TASK_SVHN_H
#define NANOCV_TASK_SVHN_H

#include "task.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // SVHN task:
        //      - digit classification
        //      - 32x32 color images as inputs
        //      - 10 outputs (10 labels)
        //
        // http://ufldl.stanford.edu/housenumbers/
        ////////////////////////////////////////////////////////////////////////////////
	
        class svhn_task_t : public task_t
        {
        public:
                // constructor
                svhn_task_t(const string_t& /*params*/ = string_t()) {}

                NCV_MAKE_CLONABLE(svhn_task_t, task_t, "SVHN (object classification)")

                // load images from the given directory
                virtual bool load(const string_t& dir);

                // access functions
                virtual size_t n_rows() const { return 32; }
                virtual size_t n_cols() const { return 32; }
                virtual size_t n_outputs() const { return 10; }                
                virtual color_mode color() const { return color_mode::rgba; }

        private:

                // load binary file
                size_t load(const string_t& bfile, protocol p);

                // decode the uncompressed bytes (images + labels)
                size_t decode(const std::vector<u_int8_t>& image_data,
                              const std::vector<u_int8_t>& label_data,
                              protocol p);

                // build folds
                bool build_folds(size_t n_train, size_t n_test);
        };
}

#endif // NANOCV_TASK_SVHN_H
