#ifndef NANOCV_TASK_CIFAR10_H
#define NANOCV_TASK_CIFAR10_H

#include "ncv_task.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // CIFAR10 task:
        //      - object classification
        //      - 32x32 color images as inputs
        //      - 10 outputs (10 labels)
        ////////////////////////////////////////////////////////////////////////////////
	
        class cifar10_task_t : public task_t
        {
        public:
                // constructor
                cifar10_task_t(const string_t& params = string_t());

                // create an object clone
                virtual rtask_t clone(const string_t& params) const
                {
                        return rtask_t(new cifar10_task_t(params));
                }

                // load images from the given directory
                virtual bool load(const string_t& dir);

                // access functions
                virtual size_t n_rows() const { return 32; }
                virtual size_t n_cols() const { return 32; }
                virtual size_t n_outputs() const { return 10; }

        private:

                // load binary file
                size_t load(const string_t& bfile, protocol p);

                // build folds
                bool build_folds(size_t n_train, size_t n_test);
        };
}

#endif // NANOCV_TASK_CIFAR10_H
