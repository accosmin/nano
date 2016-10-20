#pragma once

#include "io/buffer.h"
#include "task_mem_vision.h"

namespace nano
{
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

                virtual rtask_t clone() const override;

        private:

                virtual bool populate() override;

                // load binary file
                size_t load_binary(const string_t& bfile, const protocol p);

                // decode the uncompressed bytes (images + labels)
                size_t decode(const buffer_t& image_data, const buffer_t& label_data, const protocol p);
        };
}

