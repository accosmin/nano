#pragma once

#include "color.h"
#include "libnanocv/string.h"

namespace ncv
{
        ///
        /// \brief load RGBA image from disk
        ///
        bool load_rgba_image(const string_t& path, rgba_matrix_t& rgba);

        ///
        /// \brief load RGBA image from (memory) buffer
        ///
        bool load_rgba_image(const string_t& name, const char* buffer, size_t buffer_size, rgba_matrix_t& rgba);

        ///
        /// \brief load grayscale image from disk
        ///
        bool load_luma_image(const string_t& path, luma_matrix_t& luma);

        ///
        /// \brief load grayscale image from (memory) buffer
        ///
        bool load_luma_image(const string_t& name, const char* buffer, size_t buffer_size, luma_matrix_t& luma);

        ///
        /// \brief save RGBA image to disk
        ///
        bool save_rgba_image(const string_t& path, const rgba_matrix_t& rgba);

        ///
        /// \brief save grayscale image to disk
        ///
        bool save_luma_image(const string_t& path, const luma_matrix_t& luma);
}
