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
        /// \brief load gray/color image from [0, 1] normalized tensor
        ///
        bool load_rgba(const tensor_t& tensor, rgba_matrix_t& rgba);

        ///
        /// \brief load gray image from buffer
        ///
        bool load_gray(const char* buffer, size_t rows, size_t cols,                    // rows * cols
                       rgba_matrix_t& rgba);

        ///
        /// \brief load RGBA image from buffer
        ///
        bool load_rgba(const char* buffer, size_t rows, size_t cols, size_t stride,     // rows * cols * 3
                       rgba_matrix_t& rgba);

        namespace detail
        {
                ///
                /// \brief retrieve the [0, 1] normalized color channel
                ///
                template
                <
                        typename toperator
                >
                matrix_t make_data(const rgba_matrix_t& rgba, const rect_t& region, const toperator& op)
                {
                        const coord_t top = geom::top(region), left = geom::left(region);
                        const coord_t rows = geom::rows(region), cols = geom::cols(region);

                        matrix_t data(rows, cols);
                        for (coord_t r = 0; r < rows; r ++)
                        {
                                for (coord_t c = 0; c < cols; c ++)
                                {
                                        data(r, c) = op(rgba(top + r, left + c));
                                }
                        }
                        data /= 255.0;

                        return data;
                }
        }

        ///
        /// \brief retrieve the scaled [0, 1] RGB input vector
        ///
        inline matrix_t load_red(const rgba_matrix_t& rgba, const rect_t& region)
        {
                return detail::make_data(rgba, region, color::make_red);
        }
        inline matrix_t load_green(const rgba_matrix_t& rgba, const rect_t& region)
        {
                return detail::make_data(rgba, region, color::make_green);
        }
        inline matrix_t load_blue(const rgba_matrix_t& rgba, const rect_t& region)
        {
                return detail::make_data(rgba, region, color::make_blue);
        }
        inline matrix_t load_luma(const rgba_matrix_t& rgba, const rect_t& region)
        {
                return detail::make_data(rgba, region, color::make_luma);
        }

        ///
        /// \brief create an RGBA image composed from fixed-size RGBA patches disposed in a grid
        ///
        class grid_image_t
        {
        public:

                // constructor
                grid_image_t(   size_t patch_rows, size_t patch_cols,
                                size_t group_rows, size_t group_cols,
                                size_t border = 8,
                                rgba_t back_color = color::make_rgba(225, 225, 0));

                // setup a patch at a given grid position
                bool set(size_t grow, size_t gcol, const rgba_matrix_t& patch);

                // access functions
                const rgba_matrix_t& rgba() const { return m_image; }

        private:

                // attributes
                size_t          m_prows;        ///< patch size
                size_t          m_pcols;
                size_t          m_grows;        ///< grid size
                size_t          m_gcols;
                size_t          m_border;       ///< grid border in pixels
                rgba_t          m_bcolor;       ///< background color
                rgba_matrix_t   m_image;
        };

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

        typedef std::vector<sample_t>   samples_t;

        ///
        /// \brief compare two samples (to order them for fast caching)
        ///
        inline bool operator<(const sample_t& one, const sample_t& another)
        {
                return one.m_index < another.m_index;
        }

        typedef rgba_matrix_t           image_t;
        typedef std::vector<image_t>    images_t;
}

#endif //  NANOCV_IMAGE_H
