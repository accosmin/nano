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

                // create an object clone
                virtual rtask_t clone(const string_t& /*params*/) const
                {
                        return rtask_t(new cmufaces_task_t(*this));
                }

                // describe the object
                virtual const char* name() const { return "cmu-faces"; }
                virtual const char* desc() const { return "CMU faces (face/non-face classification)"; }

                // load images from the given directory
                virtual bool load(const string_t& dir);

                // load sample patch
                virtual void load(const image_sample_t& isample, sample_t& sample) const;

                // access functions
                virtual size_t n_rows() const { return 19; }
                virtual size_t n_cols() const { return 19; }
                virtual size_t n_inputs() const { return n_rows() * n_cols() * 1; }
                virtual size_t n_outputs() const { return 2; }

                virtual size_t n_images() const { return m_images.size(); }
                virtual const annotated_image_t& image(index_t i) const { return m_images[i]; }

                virtual size_t n_folds() const { return 1; }
                virtual const image_samples_t& fold(const fold_t& fold) const { return m_folds.find(fold)->second; }

        private:

                // load files from the given directory
                size_t load(const string_t& dir, bool is_face, protocol p);

                // build folds
                bool build_folds(size_t n_train_images, size_t n_test_images);

        private:

                // attributes
                annotated_images_t      m_images;
                fold_image_samples_t    m_folds;
        };
}

#endif // NANOCV_TASK_CMUFACES_H
