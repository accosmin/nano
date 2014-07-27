#ifndef NANOCV_TASK_NORB_H
#define NANOCV_TASK_NORB_H

#include "task.h"

namespace ncv
{
        ///
        /// NORB task:
        ///      - 3D object recognition from shape
        ///      - 108x108 grayscale images as inputs
        ///      - 5 outputs (5 labels)
        ///
        /// http://www.cs.nyu.edu/~ylclab/data/norb-v1.0/
        ///
        class norb_task_t : public task_t
        {
        public:
                
                NANOCV_MAKE_CLONABLE(norb_task_t)
                
                // constructor
                norb_task_t(const string_t& = string_t())
                        :       task_t("NORB (3D object recognition)")
                {
                }

                // load images from the given directory
                virtual bool load(const string_t& dir);

                // access functions
                virtual size_t n_rows() const { return 108; }
                virtual size_t n_cols() const { return 108; }
                virtual size_t n_outputs() const { return 5; }
                virtual size_t n_folds() const { return 1; }
                virtual color_mode color() const { return color_mode::luma; }

        private:

                // load binary file
                size_t load(const string_t& bfile, protocol p);
                size_t load(const string_t& cfile, const string_t& dfile, protocol p);
        };
}

#endif // NANOCV_TASK_NORB_H
