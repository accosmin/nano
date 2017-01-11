#pragma once

#include "color.h"

namespace nano
{
        ///
        /// \brief load image from disk as luma, RGB or RGBA (color conversions are performed if needed).
        ///
        NANO_PUBLIC bool load_luma_image(const string_t& path, image_tensor_t&);
        NANO_PUBLIC bool load_rgba_image(const string_t& path, image_tensor_t&);
        NANO_PUBLIC bool load_rgb_image(const string_t& path, image_tensor_t&);

        ///
        /// \brief load image from (memory) buffer as luma, RGB or RGBA.
        ///
        NANO_PUBLIC bool load_luma_image(const string_t& name, const char* buffer, const size_t size, image_tensor_t&);
        NANO_PUBLIC bool load_rgba_image(const string_t& name, const char* buffer, const size_t size, image_tensor_t&);
        NANO_PUBLIC bool load_rgb_image(const string_t& name, const char* buffer, const size_t size, image_tensor_t&);

        ///
        /// \brief save image to disk.
        ///
        NANO_PUBLIC bool save_image(const string_t& path, const image_tensor_t&);
}
