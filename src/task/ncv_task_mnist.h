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

                // create an object clone
                virtual rtask_t clone(const string_t& /*params*/) const
                {
                        return rtask_t(new mnist_task_t(*this));
                }

                // describe the object
                virtual const char* name() const { return "mnist"; }
                virtual const char* desc() const { return "MNIST (digit classification)"; }

                // load images from the given directory
                virtual bool load(const string_t& dir);

                // access functions
                virtual size_t n_rows() const { return 28; }
                virtual size_t n_cols() const { return 28; }
                virtual size_t n_inputs() const { return n_rows() * n_cols() * 1; }
                virtual size_t n_outputs() const { return 10; }

                virtual size_t n_images() const { return m_images.size(); }
                virtual const annotated_image_t& image(index_t i) const { return m_images[i]; }

                virtual size_t n_folds() const { return 1; }
                virtual const image_samples_t& fold(const fold_t& fold) const { return m_folds.find(fold)->second; }

        private:

                // load binary file
                size_t load(const string_t& ifile, const string_t& gfile, protocol p);

                // build folds
                bool build_folds(size_t n_train_images, size_t n_test_images);

        private:

                // attributes
                annotated_images_t      m_images;
                fold_image_samples_t    m_folds;
        };
}

#endif // NANOCV_TASK_MNIST_H
