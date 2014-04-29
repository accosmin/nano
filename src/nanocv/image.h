#ifndef  NANOCV_IMAGE_H
#define  NANOCV_IMAGE_H

#include "color.h"
#include "geom.h"

namespace ncv
{
        ///
        /// \brief save RGBA image to disk
        ///
        bool save_rgba(const string_t& path, const rgba_matrix_t& rgba);

        ///
        /// \brief load RGBA image from disk
        ///
        bool load_rgba(const string_t& path, rgba_matrix_t& rgba);

        ///
        /// \brief fold: <fold index, protocol: train|test>
        ///
        typedef std::pair<size_t, protocol>     fold_t;
        typedef std::vector<fold_t>             folds_t;

        ///
        /// \brief image-indexed sample
        ///
        struct sample_t
        {
                // constructor
                sample_t(size_t index = 0, coord_t x = 0, coord_t y = 0, coord_t w = 0, coord_t h = 0)
                        :       sample_t(index, geom::make_rect(x, y, w, h))
                {
                }
                sample_t(size_t index, const rect_t& region)
                        :       m_index(index), m_region(region),
                                m_fold{0, protocol::test}
                {
                }

                // check if this sample is annotated
                bool annotated() const { return m_target.size() > 0; }

                // attributes
                size_t          m_index;        ///< image index
                rect_t          m_region;       ///< image coordinates
                string_t        m_label;        ///< label (e.g. classification)
                vector_t        m_target;       ///< target vector to predict
                fold_t          m_fold;
        };

        typedef std::vector<sample_t>           samples_t;

        ///
        /// \brief compare two samples (to order them for fast caching)
        ///
        inline bool operator<(const sample_t& one, const sample_t& another)
        {
                return one.m_index < another.m_index;
        }

        ///
        /// \brief image with its annotations
        ///
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

                // retrieve the [0,1] normalized color channel
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

                // attributes
                rgba_matrix_t   m_rgba;
        };

        typedef std::vector<image_t>            images_t;
}

#endif //  NANOCV_IMAGE_H
