#ifndef NANOCV_IMAGE_H
#define NANOCV_IMAGE_H

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
        bool load_gray(const char* buffer, size_t rows, size_t cols, rgba_matrix_t&);
        bool load_gray(const char* buffer, size_t rows, size_t cols, gray_matrix_t&);

        ///
        /// \brief load RGBA image from buffer
        ///
        bool load_rgba(const char* buffer, size_t rows, size_t cols, size_t stride, rgba_matrix_t&);
        bool load_rgba(const char* buffer, size_t rows, size_t cols, size_t stride, gray_matrix_t&);

        namespace detail
        {
                ///
                /// \brief retrieve the [0, 1] normalized color channel
                ///
                template
                <
                        typename timage,
                        typename toperator
                >
                matrix_t make_data(const timage& image, const rect_t& region, const toperator& op)
                {
                        const coord_t top = geom::top(region), left = geom::left(region);
                        const coord_t rows = geom::rows(region), cols = geom::cols(region);
                        const scalar_t scale = 1.0 / 255.0;
                        
                        matrix_t data(rows, cols);
                        
                        if (    top == 0 && left == 0 && 
                                rows == static_cast<coord_t>(image.rows()) &&
                                cols == static_cast<coord_t>(image.cols()))
                        {
                                const coord_t size = rows * cols;
                                
                                for (coord_t i = 0; i < size; i ++)
                                {
                                        data(i) = scale * op(image(i));
                                }
                        }
                        
                        else
                        {                        
                                for (coord_t r = 0; r < rows; r ++)
                                {
                                        for (coord_t c = 0; c < cols; c ++)
                                        {
                                                data(r, c) = scale * op(image(top + r, left + c));
                                        }
                                }
                        }

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
        inline matrix_t load_luma(const gray_matrix_t& gray, const rect_t& region)
        {
                return detail::make_data(gray, region, [] (gray_t g) { return g; });
        }

        typedef rgba_matrix_t           image_t;
        typedef std::vector<image_t>    images_t;

        namespace temp
        {
                ///
                /// \brief stores an image either as grayscale or RGBA buffer.
                ///
                /// operations:
                ///     - loading and saving from and to files
                ///     - scaling to [0, 1] tensors
                ///
                /// fixme: merge grid_image here!
                ///
                class image_t
                {
                public:

                        // constructors
                        image_t(size_t rows = 0, size_t cols = 0, color_mode mode = color_mode::rgba);

                        ///
                        /// \brief save image to disk
                        ///
                        bool save_rgba(const string_t& path) const;
                        bool save_gray(const string_t& path) const;
                        bool save(const string_t& path) const;

                        ///
                        /// \brief load image from disk
                        ///
                        bool load_rgba(const string_t& path);
                        bool load_gray(const string_t& path);
                        bool load(const string_t& path);

                        ///
                        /// \brief load the scaled [0, 1] tensor
                        ///     with 1 (grayscale) or 3 (rgba) planes
                        ///
                        tensor_t as_tensor(const rect_t& region) const;

                        // access functions
                        size_t rows() const { return m_rows; }
                        size_t cols() const { return m_cols; }

                private:

                        // attributes
                        size_t                  m_rows;
                        size_t                  m_cols;
                        color_mode              m_mode;

                        rgba_matrix_t           m_rgba;
                        gray_matrix_t           m_gray;
                };
        }
}

#endif //  NANOCV_IMAGE_H
