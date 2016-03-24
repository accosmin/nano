#pragma once

#include "rect.h"
#include "color.h"
#include "stringi.h"

namespace nano
{
        ///
        /// \brief stores an image either as grayscale (luma) or RGBA buffer.
        ///
        /// operations:
        ///     - loading and saving from and to files
        ///     - scaling to [0, 1] tensors
        ///
        class NANO_PUBLIC image_t
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit image_t(const coord_t rows = 0, const coord_t cols = 0, const color_mode = color_mode::rgba);

                ///
                /// \brief resize to new dimensions
                ///
                void resize(const coord_t rows, const coord_t cols, const color_mode mode);

                ///
                /// \brief load image from disk
                ///
                bool load_rgba(const string_t& path);
                bool load_luma(const string_t& path);

                ///
                /// \brief load image from encoded buffer, using the filename's extension as a hint to the image type
                ///
                bool load_rgba(const string_t& name, const char* buffer, const size_t buffer_size);
                bool load_luma(const string_t& name, const char* buffer, const size_t buffer_size);

                ///
                /// \brief load image from decoded buffer
                ///
                bool load_luma(const char* buffer, const coord_t rows, const coord_t cols);
                bool load_rgba(const char* buffer, const coord_t rows, const coord_t cols);
                bool load_rgba(const char* buffer, const coord_t rows, const coord_t cols, const coord_t stride);
                bool load_rgba(const image_matrix_t& data);
                bool load_luma(const image_matrix_t& data);

                ///
                /// \brief save image to disk
                ///
                bool save(const string_t& path) const;

                ///
                /// \brief save image to scaled [0, 1] tensor
                ///     with 1 (luma) or 3 (rgb) planes
                ///
                tensor3d_t to_tensor() const;
                tensor3d_t to_tensor(const rect_t& region) const;

                ///
                /// \brief transform between color modes
                ///
                bool make_rgba();
                bool make_luma();

                ///
                /// \brief fill with constant color
                ///
                bool fill(const rgba_t rgba);
                bool fill(const luma_t luma);
                bool fill(const rect_t& rect, const rgba_t rgba);
                bool fill(const rect_t& rect, const luma_t luma);

                ///
                /// \brief copy the given (region of the given) patch at the (top, left) location
                ///
                bool copy(const coord_t top, const coord_t left, const rgba_matrix_t& patch);
                bool copy(const coord_t top, const coord_t left, const luma_matrix_t& patch);

                bool copy(const coord_t top, const coord_t left, const image_t& patch);
                bool copy(const coord_t top, const coord_t left, const image_t& patch, const rect_t& patch_region);

                ///
                /// \brief transpose in place the pixel matrix
                ///
                void transpose_in_place();

                ///
                /// \brief set pixels to random values
                ///
                bool random();

                ///
                /// \brief check if the given rectangle is within image bounds
                ///
                bool valid(const rect_t& rect) const
                {
                        return  rect.left() >= 0 && rect.right() <= cols() &&
                                rect.top() >= 0 && rect.bottom() <= rows();
                }

                // access functions
                coord_t dims() const { return static_cast<coord_t>(m_data.size<0>()); }
                coord_t rows() const { return static_cast<coord_t>(m_data.size<1>()); }
                coord_t cols() const { return static_cast<coord_t>(m_data.size<2>()); }
                coord_t size() const { return rows() * cols(); }

                bool is_rgba() const { return mode() == color_mode::rgba; }
                bool is_luma() const { return mode() == color_mode::luma; }
                color_mode mode() const { return m_mode; }

                const auto& data() const { return m_data; }

        private:

                // attributes
                color_mode              m_mode;
                vision_tensor_t         m_data;
        };

        using images_t = std::vector<image_t>;
}
