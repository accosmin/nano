#pragma once

#include "rect.h"
#include "color.h"
#include "stringi.h"

namespace nano
{
        ///
        /// \brief stores an image as grayscale (luma), RGB or RGBA.
        ///
        /// operations:
        ///     - loading and saving from and to files
        ///     - scaling to/from [0, 1] 3D tensors
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
                bool load_luma(const string_t& path);
                bool load_rgba(const string_t& path);
                bool load_rgb(const string_t& path);

                ///
                /// \brief load image from encoded buffer, using the filename's extension as a hint to the image type
                ///
                bool load_rgba(const string_t& name, const char* buffer, const size_t buffer_size);
                bool load_luma(const string_t& name, const char* buffer, const size_t buffer_size);
                bool load_rgb(const string_t& name, const char* buffer, const size_t buffer_size);

                ///
                /// \brief load image from decoded buffer
                ///
                bool load_luma(const char* buffer, const coord_t rows, const coord_t cols);
                bool load_rgba(const char* buffer, const coord_t rows, const coord_t cols, const coord_t stride);
                bool load_rgb(const char* buffer, const coord_t rows, const coord_t cols, const coord_t stride);

                ///
                /// \brief load from image tensor (keeps the color channels)
                ///
                bool load(const image_tensor_t& data);

                ///
                /// \brief save image to disk
                ///
                bool save(const string_t& path) const;

                ///
                /// \brief save image to scaled [0, 1] tensor
                ///
                tensor3d_t to_tensor() const;
                tensor3d_t to_tensor(const rect_t& region) const;

                ///
                /// \brief load image from scaled [0, 1] tensor
                ///
                bool from_tensor(const tensor3d_t& data);

                ///
                /// \brief transform to another color mode (if not already)
                ///
                void make_luma();
                void make_rgba();
                void make_rgb();

                ///
                /// \brief fill with constant color
                ///
                void fill(const luma_t);
                void fill(const rgba_t);
                void fill(const rgb_t);

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
                size_t hash() const;

                bool is_rgb() const { return mode() == color_mode::rgb; }
                bool is_rgba() const { return mode() == color_mode::rgba; }
                bool is_luma() const { return mode() == color_mode::luma; }
                color_mode mode() const;

                auto plane(const coord_t band) const { return m_data.matrix(band); }
                auto plane(const coord_t band) { return m_data.matrix(band); }

                auto plane(const coord_t band, const coord_t t, const coord_t l, const coord_t h, const coord_t w) const
                {
                        return plane(band).block(t, l, h, w);
                }
                auto plane(const coord_t band, const coord_t t, const coord_t l, const coord_t h, const coord_t w)
                {
                        return plane(band).block(t, l, h, w);
                }

                auto plane(const coord_t band, const rect_t& rect) const
                {
                        return plane(band, rect.top(), rect.left(), rect.height(), rect.width());
                }
                auto plane(const coord_t band, const rect_t& rect)
                {
                        return plane(band, rect.top(), rect.left(), rect.height(), rect.width());
                }

        private:

                // attributes
                image_tensor_t          m_data;
        };

        using images_t = std::vector<image_t>;
}
