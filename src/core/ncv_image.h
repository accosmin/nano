#ifndef  NANOCV_IMAGE_H
#define  NANOCV_IMAGE_H

#include "ncv_color.h"
#include "ncv_geom.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // save/load RGBA image to disk
        ////////////////////////////////////////////////////////////////////////////////

        bool save_rgba(const string_t& path, const rgba_matrix_t& rgba);
        bool load_rgba(const string_t& path, rgba_matrix_t& rgba);

        ////////////////////////////////////////////////////////////////////////////////
        // image annotation
        ////////////////////////////////////////////////////////////////////////////////

        struct annotation_t
        {
                // constructor
                annotation_t(coord_t x = 0, coord_t y = 0, coord_t w = 0, coord_t h = 0,
                           const string_t& label = string_t(),
                           const vector_t& target = vector_t())
                        :       annotation_t(geom::make_rect(x, y, w, h), label, target)
                {
                }

                annotation_t(const rect_t& region, const string_t& label, const vector_t& target)
                        :       m_region(region),
                                m_label(label),
                                m_target(target)
                {
                }

                // attributes
                rect_t          m_region;       // 2D annotated region
                string_t        m_label;        // label (e.g. classification)
                vector_t        m_target;       // target vector to predict
        };

        typedef std::vector<annotation_t>       annotations_t;

        ////////////////////////////////////////////////////////////////////////////////
        // image-indexed sample
        ////////////////////////////////////////////////////////////////////////////////

        struct isample_t
        {
                // constructor
                isample_t(size_t index = 0, coord_t x = 0, coord_t y = 0, coord_t w = 0, coord_t h = 0)
                        :       isample_t(index, geom::make_rect(x, y, w, h))
                {
                }
                isample_t(size_t index, const rect_t& region)
                        :       m_index(index), m_region(region)
                {
                }

                // attributes
                size_t          m_index;        // image index
                rect_t          m_region;       // image coordinates
        };

        typedef std::vector<isample_t>          isamples_t;

        // fold image-indexed samples
        typedef std::pair<size_t, protocol>     fold_t;
        typedef std::map<fold_t, isamples_t>    folds_t;

        ////////////////////////////////////////////////////////////////////////////////
        // image with its annotations
        ////////////////////////////////////////////////////////////////////////////////

        struct image_t
        {
                // load gray/color image from buffer
                bool load(const string_t& path);
                bool load_gray(const char* buffer, size_t rows, size_t cols);
                bool load_rgba(const char* buffer, size_t rows, size_t cols);

                // retrieve the scaled to [0, 1] RGB input vector & the associated target (if any)
                vector_t get_input(const rect_t& region) const;
                vector_t get_target(const rect_t& region) const;

                // check if a target is valid
                static bool has_target(const vector_t& target) { return target.size() > 0; }

                // attributes
                rgba_matrix_t   m_rgba;
                annotations_t   m_annotations;
                protocol        m_protocol;
        };

        typedef std::vector<image_t>            images_t;
}

#endif //  NANOCV_IMAGE_H
