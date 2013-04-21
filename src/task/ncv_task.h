#ifndef NANOCV_TASK_H
#define NANOCV_TASK_H

#include "ncv_manager.h"
#include "ncv_color.h"
#include "ncv_sample.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // describes a generic computer vision task consisting
        //      of a set of images with annotations
        //      and a protocol (training + validation + testing).
        //
        // samples for training & testing models can be drawn from these image.
	////////////////////////////////////////////////////////////////////////////////
	
        class task : public clonable<task>
	{
        public:
                
                // destructor
                virtual ~task() {}
                
                // load data from disk
                virtual bool load(const string_t& dir) = 0;

                // access functions
                virtual size_t n_images() const = 0;
                virtual size_t n_images(protocol dtype) const = 0;
                virtual const cielab_matrix_t& image(protocol dtype, index_t i) const = 0;
                
                virtual size_t n_samples() const = 0;
                virtual size_t n_samples(protocol dtype) const = 0;
                virtual sample operator()(protocol dtype, index_t i) const = 0;
                
                virtual size_t n_labels() const = 0;
                virtual const strings_t& labels() const = 0;
        };
}

#endif // NANOCV_TASK_H
