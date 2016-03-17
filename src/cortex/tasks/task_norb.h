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

                ///
                /// \brief short name of this task
                ///
                virtual string_t name() const override { return "norb"; }

                ///
                /// \brief load the task from the given directory (if possible)
                ///
                virtual bool load(const string_t& dir = string_t()) override;

        private:

                // load binary file
                bool load(const string_t& bfile, protocol p, size_t count);
                bool load(const string_t& ifile, const string_t& gfile, protocol p, size_t count);
        };
}

