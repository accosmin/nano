#ifndef NANOCV_TASK_H
#define NANOCV_TASK_H

#include "ncv_manager.h"
#include "ncv_sample.h"

namespace ncv
{
        // manage the losses (register new ones, query and clone them)
        class task;
        typedef manager<task>                   task_manager;
        typedef task_manager::robject_t         rtask;

        ////////////////////////////////////////////////////////////////////////////////
        // describes a generic computer vision task consisting
        //      of a set of images with annotations
        //      and a protocol (training + validation + testing).
        // samples for training & testing models can be drawn from these image.
	////////////////////////////////////////////////////////////////////////////////
	
        class task : public clonable<task>
	{
        public:
                
                // destructor
                virtual ~task() {}
                
                // load samples from disk to fit in the given memory amount
                virtual bool load(
                        const string_t& dir,
                        size_t ram_gb,
                        samples_t& train_samples,
                        samples_t& valid_samples,
                        samples_t& test_samples) = 0;

                // access functions
                virtual size_t n_labels() const = 0;
                virtual const strings_t& labels() const = 0;
        };
}

#endif // NANOCV_TASK_H
