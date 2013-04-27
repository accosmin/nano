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
	
        class stl10_task : public task
        {
        public:
                // create an object clone
                virtual rtask clone(const string_t& /*params*/) const
                {
                        return rtask(new stl10_task(*this));
                }

                // describe the object
                virtual const char* name() const { return "stl10"; }
                virtual const char* desc() const { return "stl 10 (object classification)"; }

                // load images from the given directory
                virtual bool load(const string_t& dir);

                // sample training & testing samples
                virtual size_t n_folds() const { return 10; }
                virtual size_t fold_size(index_t f, protocol p) const;
                virtual bool fold_sample(index_t f, protocol p, index_t s, sample& ss) const;

                // access functions
                virtual size_t n_rows() const { return 96; }
                virtual size_t n_cols() const { return 96; }
                virtual size_t n_inputs() const { return n_rows() * n_cols() * 3; }
                virtual size_t n_outputs() const { return 10; }

                virtual size_t n_images() const { return m_images.size(); }
                virtual const annotated_image& image(index_t i) const { return m_images[i]; }
                                                   
        private:
                                                   
                // load binary file
                size_t load(const string_t& ifile, const string_t& gfile, protocol p);
                size_t load(const string_t& ifile, protocol p);

        private:

                // attributes
                annotated_images_t      m_images;
        };
}

#endif // NANOCV_TASK_STL10_H
