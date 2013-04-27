#ifndef NANOCV_TASK_H
#define NANOCV_TASK_H

#include "ncv_manager.h"
#include "ncv_sample.h"

namespace ncv
{
        // manage tasks (register new ones, query and clone them)
        class task;
        typedef manager<task>                   task_manager;
        typedef task_manager::robject_t         rtask;

        ////////////////////////////////////////////////////////////////////////////////
        // generic computer vision task consisting of a set of (annotated) images
        //      and a protocol (training + testing).
        // samples for training & testing models can be drawn from these image.
	////////////////////////////////////////////////////////////////////////////////
	
        class task : public clonable<task>
	{
        public:
                
                // destructor
                virtual ~task() {}

                // load images from the given directory
                virtual bool load(const string_t& dir) = 0;

                // sample training & testing samples
                virtual size_t n_folds() const = 0;
                virtual size_t fold_size(index_t f, protocol p) const = 0;
                virtual bool fold_sample(index_t f, protocol p, index_t s, sample& ss) const = 0;

                // access functions
                virtual size_t n_rows() const = 0;
                virtual size_t n_cols() const = 0;
                virtual size_t n_inputs() const = 0;
                virtual size_t n_outputs() const = 0;

                virtual size_t n_images() const = 0;
                virtual const annotated_image& image(index_t i) const = 0;
        };
}

#endif // NANOCV_TASK_H
