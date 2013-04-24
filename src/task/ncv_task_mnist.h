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
	
        class mnist_task : public task
        {
        public:
                
                // constructor
                mnist_task();

                // destructor
                virtual ~mnist_task();

                // load samples from disk to fit in the given memory amount
                virtual bool load(
                        const string_t& dir,
                        size_t ram_gb,
                        samples_t& train_samples,
                        samples_t& valid_samples,
                        samples_t& test_samples);

                // create an object clone
                virtual rtask clone(const string_t& /*params*/) const
                {
                        return rtask(new mnist_task(*this));
                }

                // describe the object
                virtual const char* name() const { return "mnist"; }
                virtual const char* desc() const { return "mnist (digit classification)"; }

                // access functions
                virtual size_t n_rows() const { return 28; }
                virtual size_t n_cols() const { return 28; }
                virtual size_t n_outputs() const { return 10; }
                virtual size_t n_labels() const { return m_labels.size(); }
                virtual const strings_t& labels() const { return m_labels; }
                                                   
        private:
                                                   
                // read binary file into either <dtype1> or <dtype2> datasets with the given probability
                bool load(const string_t& ifile, const string_t& gfile,
                        samples_t& samples1, samples_t& samples2, scalar_t prob);

        private:

                // attributes
                strings_t       m_labels;
        };
}

#endif // NANOCV_TASK_MNIST_H
