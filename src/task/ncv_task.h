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

                // constructor
                task_t(const string_t& name, const string_t& description)
                        :       clonable_t<task_t>(name, description)
                {
                }
                
                // destructor
                virtual ~task_t() {}

                // load images from the given directory
                virtual bool load(const string_t& dir) = 0;

                // load sample patch
                virtual void load(const isample_t& isample, sample_t& sample) const = 0;

                // sample size
                rect_t sample_size() const;

                // sample region at a particular offset
                rect_t sample_region(coord_t x, coord_t y) const;

                // access functions
                virtual size_t n_rows() const = 0;
                virtual size_t n_cols() const = 0;
                virtual size_t n_inputs() const = 0;
                virtual size_t n_outputs() const = 0;

                size_t n_images() const { return m_images.size(); }
                const image_t& image(size_t i) const { return m_images[i]; }

                size_t n_folds() const { return m_folds.size() / 2; } // train + test
                const isamples_t& fold(const fold_t& fold) const { return m_folds.find(fold)->second; }

        protected:

                // construct image-indexed samples for the [istart, istart + icount) images
                //      having the (region) image coordinates
                isamples_t make_isamples(size_t istart, size_t icount, const rect_t& region);

        protected:

                // attributes
                images_t                m_images;
                folds_t                 m_folds;
        };
}

#endif // NANOCV_TASK_H
