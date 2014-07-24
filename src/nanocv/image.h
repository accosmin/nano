#ifndef NANOCV_IMAGE_H
#define NANOCV_IMAGE_H

#include "color.h"
#include "geom.h"

namespace ncv
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
                /// \brief resize to new dimensions
                ///
                void resize(size_t rows, size_t cols, color_mode mode);

                ///
                /// \brief load image from disk
                ///
                bool load_rgba(const string_t& path);
                bool load_gray(const string_t& path);
                bool load(const string_t& path);

                ///
                /// \brief load image from buffer
                ///
                bool load_gray(const char* buffer, size_t rows, size_t cols);
                bool load_rgba(const char* buffer, size_t rows, size_t cols);
                bool load_rgba(const char* buffer, size_t rows, size_t cols, size_t stride);

                ///
                /// \brief load image from scaled [0, 1] tensor
                ///     having 1 (grayscale) or 3 (rgba) planes
                ///
                bool from_tensor(const tensor_t& data) const;
                bool from_tensor(const tensor_t& data, const rect_t& region) const;

                ///
                /// \brief save image to disk
                ///
                bool save_rgba(const string_t& path) const;
                bool save_gray(const string_t& path) const;
                bool save(const string_t& path) const;

                ///
                /// \brief save image to scaled [0, 1] tensor
                ///     with 1 (grayscale) or 3 (rgba) planes
                ///
                tensor_t to_tensor() const;
                tensor_t to_tensor(const rect_t& region) const;

                ///
                /// \brief fill image with constant color
                ///
                void fill(rgba_t rgba) const;
                void fill(gray_t gray) const;

                ///
                /// \brief fill image region with the given patch
                ///
                bool fill(const rect_t& region, const rgba_matrix_t& data) const;
                bool full(const rect_t& region, const gray_matrix_t& data) const;

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

        typedef std::vector<image_t>    images_t;
}

#endif //  NANOCV_IMAGE_H
