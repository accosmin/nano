#ifndef NANOCV_TASK_MNIST_H
#define NANOCV_TASK_MNIST_H

#include "ncv_task.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // MNIST task:
        //      - digit classification
        //      - 28x28 grayscale images as inputs
        //      - 10 outputs (10 labels)
        ////////////////////////////////////////////////////////////////////////////////
	
        class mnist_task_t : public task_t
        {
        public:
                // constructor
                mnist_task_t(const string_t& params = string_t());

                // create an object clone
                virtual rtask_t clone(const string_t& params) const
                {
                        return rtask_t(new mnist_task_t(params));
                }

                // load images from the given directory
                virtual bool load(const string_t& dir);

                // access functions
                virtual size_t n_rows() const { return 28; }
                virtual size_t n_cols() const { return 28; }
                virtual size_t n_outputs() const { return 10; }

        private:

                // load binary file
                size_t load(const string_t& ifile, const string_t& gfile, protocol p);

                // build folds
                bool build_folds(size_t n_train, size_t n_test);
        };
}

#endif // NANOCV_TASK_MNIST_H
