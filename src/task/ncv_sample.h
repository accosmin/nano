#ifndef NANOCV_SAMPLE_H
#define NANOCV_SAMPLE_H

#include "ncv_annotation.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // image-indexed sample
        ////////////////////////////////////////////////////////////////////////////////

        struct isample_t
        {
                // constructor
                isample_t(index_t index = 0, icoord_t x = 0, icoord_t y = 0, icoord_t w = 0, icoord_t h = 0)
                        :       isample_t(index, make_rect(x, y, w, h))
                {
                }
                isample_t(index_t index, const irect_t& region)
                        :       m_index(index), m_region(region)
                {
                }

                // attributes
                index_t         m_index;        // image index
                irect_t         m_region;       // image coordinates
        };

        typedef std::vector<isample_t>          isamples_t;

        // construct image-indexed samples for the [istart, istart + icount) images
        //      having the (region) image coordinates
        isamples_t make_isamples(index_t istart, index_t icount, const irect_t& region);

        // fold image-indexed samples
        typedef std::pair<index_t, protocol>    fold_t;
        typedef std::map<fold_t, isamples_t>    fold_isamples_t;

        ////////////////////////////////////////////////////////////////////////////////
        // sample data
        ////////////////////////////////////////////////////////////////////////////////

        struct sample_t
        {
                // check if annotated
                bool has_annotation() const { return m_target.size() > 0; }

                // load from gray/color image
                void load_gray(const annotated_image_t& aimage, const isample_t& isample);
                void load_rgba(const annotated_image_t& aimage, const isample_t& isample);

                // load target
                void load_target(const annotated_image_t& aimage, const isample_t& isample);

                // attributes
                vector_t        m_data;
                vector_t        m_target;       // target vector (if empty then no annotation is given)
        };

        typedef std::vector<sample_t>           samples_t;
}

#endif // NANOCV_SAMPLE_H
