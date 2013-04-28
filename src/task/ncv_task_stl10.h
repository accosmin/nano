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
                // create an object clone
                virtual rtask_t clone(const string_t& /*params*/) const
                {
                        return rtask_t(new stl10_task_t(*this));
                }

                // describe the object
                virtual const char* name() const { return "stl10"; }
                virtual const char* desc() const { return "stl 10 (object classification)"; }

                // load images from the given directory
                virtual bool load(const string_t& dir);

                // access functions
                virtual size_t n_rows() const { return 96; }
                virtual size_t n_cols() const { return 96; }
                virtual size_t n_inputs() const { return n_rows() * n_cols() * 3; }
                virtual size_t n_outputs() const { return 10; }

                virtual size_t n_images() const { return m_images.size(); }
                virtual const annotated_image_t& image(index_t i) const { return m_images[i]; }

                virtual size_t n_folds() const { return 10; }
                virtual const image_samples_t& fold(const fold_t& fold) const { return m_folds.find(fold)->second; }
                                                   
        private:
                                                   
                // load binary file
                size_t load(const string_t& ifile, const string_t& gfile, protocol p);
                size_t load(const string_t& ifile, protocol p);

                // build folds
                bool build_folds(const string_t& ifile,
                                 size_t n_train_images, size_t n_unlabeled_images, size_t n_test_images);

        private:

                // attributes
                annotated_images_t      m_images;
                fold_image_samples_t    m_folds;
        };
}

#endif // NANOCV_TASK_STL10_H
