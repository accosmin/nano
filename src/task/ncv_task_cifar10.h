#ifndef NANOCV_TASK_CIFAR10_H
#define NANOCV_TASK_CIFAR10_H

#include "ncv_task_class.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // CIFAR10 task:
        //      - object classification
        //      - 32x32 color images as inputs
        //      - 10 outputs (10 labels)
        ////////////////////////////////////////////////////////////////////////////////
	
        class cifar10_task : public task
        {
        public:
                
                // constructor
                cifar10_task();

                // destructor
                virtual ~cifar10_task();

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
                        return rtask(new cifar10_task(*this));
                }

                // describe the object
                virtual const char* name() const { return "cifar10"; }
                virtual const char* desc() const { return "cifar 10 (object classification)"; }

                // access functions
                virtual size_t n_rows() const { return 32; }
                virtual size_t n_cols() const { return 32; }
                virtual size_t n_outputs() const { return 10; }
                virtual size_t n_labels() const { return m_labels.size(); }
                virtual const strings_t& labels() const { return m_labels; }

        private:

                // read binary file into either <dtype1> or <dtype2> datasets with the given probability
                bool load(const string_t& bfile,
                        samples_t& samples1, samples_t& samples2, scalar_t prob);

        private:

                // attributes
                strings_t       m_labels;
        };
}

#endif // NANOCV_TASK_CIFAR10_H
