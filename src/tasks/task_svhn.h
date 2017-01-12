#pragma once

#include "io/buffer.h"
#include "task_mem_vision.h"

namespace nano
{
        class zlib_istream_t;

        ///
        /// SVHN task:
        ///      - digit classification
        ///      - 32x32 color images as inputs
        ///      - 10 outputs (10 labels)
        ///
        /// http://ufldl.stanford.edu/housenumbers/
        ///
        class svhn_task_t final : public mem_vision_task_t
        {
        public:

                explicit svhn_task_t(const string_t& configuration = string_t());

        private:

                virtual bool populate() override;

                size_t load_binary(const string_t& bfile, const protocol p);
                size_t load_images(zlib_istream_t&, const protocol p);
                size_t load_labels(zlib_istream_t&, size_t image_index, const protocol p);
        };
}

