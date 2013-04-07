#ifndef  NANOCV_IMAGE_H
#define  NANOCV_IMAGE_H

#include "ncv_color.h"

namespace ncv
{
        // Save/load RGBA image to disk
        bool save_image(const string_t& path, const rgba_matrix_t& rgba);
        bool load_image(const string_t& path, rgba_matrix_t& rgba);
}

#endif //  NANOCV_IMAGE_H
