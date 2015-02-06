#pragma once

#include "color.h"

namespace ncv
{
        bool load_rgba_image(const string_t& path, rgba_matrix_t& rgba);
        bool load_rgba_image(const string_t& name, const char* buffer, size_t buffer_size, rgba_matrix_t& rgba);

        bool load_luma_image(const string_t& path, luma_matrix_t& luma);
        bool load_luma_image(const string_t& name, const char* buffer, size_t buffer_size, luma_matrix_t& luma);

        bool save_rgba_image(const string_t& path, const rgba_matrix_t& rgba);
        bool save_luma_image(const string_t& path, const luma_matrix_t& luma);
}
