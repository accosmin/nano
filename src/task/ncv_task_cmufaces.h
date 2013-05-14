#ifndef NANOCV_TASK_CMUFACES_H
#define NANOCV_TASK_CMUFACES_H

#include "ncv_task.h"

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
                cmufaces_task_t(const string_t& params = string_t());

                // create an object clone
                virtual rtask_t clone(const string_t& params) const
                {
                        return rtask_t(new cmufaces_task_t(params));
                }

                // load images from the given directory
                virtual bool load(const string_t& dir);

                // load samples
                virtual void load(const fold_t& fold, samples_t& samples) const;
                virtual sample_t load(const isample_t& isample) const;

                // access functions
                virtual size_t n_rows() const { return 19; }
                virtual size_t n_cols() const { return 19; }
                virtual size_t n_inputs() const { return n_rows() * n_cols() * 1; }
                virtual size_t n_outputs() const { return 2; }

        private:

                // load files from the given directory
                size_t load(const string_t& dir, bool is_face, protocol p);

                // build folds
                bool build_folds(size_t n_train, size_t n_test);
        };
}

#endif // NANOCV_TASK_CMUFACES_H
