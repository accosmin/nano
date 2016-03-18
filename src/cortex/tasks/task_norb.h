#pragma once

#include "task_mem_vision.h"

namespace nano
{
        ///
        /// NORB task:
        ///      - 3D object recognition from shape
        ///      - 108x108 grayscale images as inputs
        ///      - 5 outputs (5 labels)
        ///
        /// http://www.cs.nyu.edu/~ylclab/data/norb-v1.0/
        ///
        class norb_task_t : public mem_vision_task_t
        {
        public:

                NANO_MAKE_CLONABLE(norb_task_t, "NORB (3D object recognition)")

                ///
                /// \brief constructor
                ///
                explicit norb_task_t(const string_t& configuration = string_t());

        private:

                virtual bool populate(const string_t& dir) override;

                // load binary file
                bool load_binary(const string_t& bfile, const protocol p, const size_t count);
                bool load_binary(const string_t& ifile, const string_t& gfile, protocol p, const size_t count);
        };
}

