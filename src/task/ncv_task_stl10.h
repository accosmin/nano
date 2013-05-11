#ifndef NANOCV_TASK_STL10_H
#define NANOCV_TASK_STL10_H

#include "ncv_task.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // STL10 task:
        //      - object classification
        //      - 96x96 color images as inputs
        //      - 10 outputs (10 labels)
        //
        // http://www.stanford.edu/~acoates//stl10/
        ////////////////////////////////////////////////////////////////////////////////
	
        class stl10_task_t : public task_t
        {
        public:
                // constructor
                stl10_task_t(const string_t& params = string_t());

                // create an object clone
                virtual rtask_t clone(const string_t& params) const
                {
                        return rtask_t(new stl10_task_t(params));
                }

                // load images from the given directory
                virtual bool load(const string_t& dir);

                // load samples
                virtual void load(const fold_t& fold, samples_t& samples) const;

                // access functions
                virtual size_t n_rows() const { return 96; }
                virtual size_t n_cols() const { return 96; }
                virtual size_t n_inputs() const { return n_rows() * n_cols() * 3; }
                virtual size_t n_outputs() const { return 10; }
                                                   
        private:
                                                   
                // load binary file
                size_t load(const string_t& ifile, const string_t& gfile, protocol p);
                size_t load(const string_t& ifile, protocol p);

                // build folds
                bool build_folds(const string_t& ifile, size_t n_train, size_t n_unlabeled, size_t n_test);
        };
}

#endif // NANOCV_TASK_STL10_H
