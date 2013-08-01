#ifndef NANOCV_TASK_CMUFACES_H
#define NANOCV_TASK_CMUFACES_H

#include "task.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // CMU-faces task:
        //      - face/non-face classification
        //      - 19x19 grayscale images as inputs
        //      - 2 outputs (binary classification)
        ////////////////////////////////////////////////////////////////////////////////
	
        class cmufaces_task_t : public task_t
        {
        public:

                // constructor
                cmufaces_task_t(const string_t& /*params*/ = string_t()) {}

                NCV_MAKE_CLONABLE(cmufaces_task_t, task_t, "CMU faces (face/non-face classification)")

                // load images from the given directory
                virtual bool load(const string_t& dir);

                // access functions
                virtual size_t n_rows() const { return 19; }
                virtual size_t n_cols() const { return 19; }
                virtual size_t n_outputs() const { return 2; }
                virtual color_mode color() const { return color_mode::luma; }

        private:

                // load files from the given directory
                size_t load(const string_t& dir, bool is_face, protocol p);

                // build folds
                bool build_folds(size_t n_train, size_t n_test);
        };
}

#endif // NANOCV_TASK_CMUFACES_H
