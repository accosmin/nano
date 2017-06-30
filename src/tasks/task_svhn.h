#pragma once

#include "task_mem_vision.h"

namespace nano
{
        class istream_t;
        struct mat5_section_t;

        ///
        /// SVHN task:
        ///      - digit classification
        ///      - 32x32 color images as inputs
        ///      - 10 outputs (10 labels)
        ///
        /// http://ufldl.stanford.edu/housenumbers/
        ///
        struct svhn_task_t final : public mem_vision_task_t
        {
                explicit svhn_task_t(const string_t& params = string_t());

                virtual bool populate() override;

                tensor_size_t load_binary(const string_t& path, const protocol);
                tensor_size_t load_pixels(const mat5_section_t&, const string_t&, const std::vector<int32_t>&, istream_t&);
                tensor_size_t load_labels(const mat5_section_t&, const string_t&, const std::vector<int32_t>&, const protocol, istream_t&);
        };
}

