#pragma once

#include "color.h"
#include "stringi.h"

namespace nano
{
        ///
        /// \brief load RGBA image from disk
        ///
        NANO_PUBLIC bool load_rgba_image(const string_t& path, image_matrix_t&);

        ///
        /// \brief load RGBA image from (memory) buffer
        ///
        NANO_PUBLIC bool load_rgba_image(const string_t& name, const char* buffer, const size_t buffer_size, image_matrix_t&);

        ///
        /// \brief load grayscale image from disk
        ///
        NANO_PUBLIC bool load_luma_image(const string_t& path, image_matrix_t&);

        ///
        /// \brief load grayscale image from (memory) buffer
        ///
        NANO_PUBLIC bool load_luma_image(const string_t& name, const char* buffer, const size_t buffer_size, image_matrix_t&);

        ///
        /// \brief save RGBA image to disk
        ///
        NANO_PUBLIC bool save_rgba_image(const string_t& path, const image_matrix_t&);

        ///
        /// \brief save grayscale image to disk
        ///
        NANO_PUBLIC bool save_luma_image(const string_t& path, const image_matrix_t&);
}
