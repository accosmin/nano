#pragma once

#include "color.h"
#include "stringi.h"

namespace cortex
{
        ///
        /// \brief load RGBA image from disk
        ///
        NANOCV_PUBLIC bool load_rgba_image(const string_t& path, rgba_matrix_t& rgba);

        ///
        /// \brief load RGBA image from (memory) buffer
        ///
        NANOCV_PUBLIC bool load_rgba_image(const string_t& name, const char* buffer, size_t buffer_size, rgba_matrix_t&);

        ///
        /// \brief load grayscale image from disk
        ///
        NANOCV_PUBLIC bool load_luma_image(const string_t& path, luma_matrix_t& luma);

        ///
        /// \brief load grayscale image from (memory) buffer
        ///
        NANOCV_PUBLIC bool load_luma_image(const string_t& name, const char* buffer, size_t buffer_size, luma_matrix_t&);

        ///
        /// \brief save RGBA image to disk
        ///
        NANOCV_PUBLIC bool save_rgba_image(const string_t& path, const rgba_matrix_t& rgba);

        ///
        /// \brief save grayscale image to disk
        ///
        NANOCV_PUBLIC bool save_luma_image(const string_t& path, const luma_matrix_t& luma);
}
