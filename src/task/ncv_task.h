#ifndef NANOCV_TASK_H
#define NANOCV_TASK_H

#include "ncv_manager.h"
#include "ncv_sample.h"

namespace ncv
{
        // manage tasks (register new ones, query and clone them)
        class task_t;
        typedef manager_t<task_t>               task_manager_t;
        typedef task_manager_t::robject_t       rtask_t;

        ////////////////////////////////////////////////////////////////////////////////
        // generic computer vision task consisting of a set of (annotated) images
        //      and a protocol (training + testing).
        // samples for training & testing models can be drawn from these image.
	////////////////////////////////////////////////////////////////////////////////
	
        class task_t : public clonable_t<task_t>
	{
        public:
                
                // destructor
                virtual ~task_t() {}

                // load images from the given directory
                virtual bool load(const string_t& dir) = 0;

                // load sample patch
                virtual void load(const isample_t& isample, sample_t& sample) const = 0;

                // access functions
                virtual size_t n_rows() const = 0;
                virtual size_t n_cols() const = 0;
                virtual size_t n_inputs() const = 0;
                virtual size_t n_outputs() const = 0;
                irect_t region() const { return make_rect(0, 0, n_cols(), n_rows()); }

                virtual size_t n_images() const = 0;
                virtual const annotated_image_t& image(index_t i) const = 0;

                virtual size_t n_folds() const = 0;
                virtual const isamples_t& fold(const fold_t& fold) const = 0;
        };
}

#endif // NANOCV_TASK_H
