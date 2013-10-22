#ifndef  NANOCV_IMAGE_H
#define  NANOCV_IMAGE_H

#include "color.h"
#include "geom.h"
#include <map>

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

        struct sample_t
        {
                // constructor
                sample_t(size_t index = 0, coord_t x = 0, coord_t y = 0, coord_t w = 0, coord_t h = 0)
                        :       sample_t(index, geom::make_rect(x, y, w, h))
                {
                }
                sample_t(size_t index, const rect_t& region)
                        :       m_index(index), m_region(region)
                {
                }

                // attributes
                size_t          m_index;        // image index
                rect_t          m_region;       // image coordinates
        };

        typedef std::vector<sample_t>           samples_t;

        // fold image-indexed samples
        typedef std::pair<size_t, protocol>     fold_t;
        typedef std::map<fold_t, samples_t>    folds_t;

        ////////////////////////////////////////////////////////////////////////////////
        // image with its annotations
        ////////////////////////////////////////////////////////////////////////////////

        struct image_t
        {
                // load gray/color image from file or buffer
                bool load(const string_t& path);
                bool load_gray(const char* buffer, size_t rows, size_t cols);                   // rows * cols
                bool load_rgba(const char* buffer, size_t rows, size_t cols, size_t stride);    // rows * cols * 3

                // retrieve the scaled [0, 1] RGB input vector
                matrix_t make_red(const rect_t& region) const { return get(region, color::make_red); }
                matrix_t make_green(const rect_t& region) const { return get(region, color::make_green); }
                matrix_t make_blue(const rect_t& region) const { return get(region, color::make_blue); }
                matrix_t make_luma(const rect_t& region) const { return get(region, color::make_luma); }

                // retrieve the associated target (if any)
                vector_t make_target(const rect_t& region) const;

                template
                <
                        typename toperator
                >
                matrix_t get(const rect_t& region, const toperator& op) const
                {
                        const coord_t top = geom::top(region), left = geom::left(region);
                        const coord_t rows = geom::rows(region), cols = geom::cols(region);

                        matrix_t data(rows, cols);
                        for (coord_t r = 0; r < rows; r ++)
                        {
                                for (coord_t c = 0; c < cols; c ++)
                                {
                                        const rgba_t rgba = m_rgba(top + r, left + c);
                                        data(r, c) = op(rgba);
                                }
                        }
                        data /= 255.0;

                        return data;
                }

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
