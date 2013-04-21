#include "ncv_tensor.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        tensor_data_t tensor::make(size_t rows, size_t cols, size_t i)
        {
                tensor_data_t data(rows, cols);
                math::for_each(data, [&] (vector_t& v)
                {
                        v.resize(i);
                });

                return data;
        }

        //-------------------------------------------------------------------------------------------------

        tensor_kernel_t tensor::make(size_t rows, size_t cols, size_t o, size_t i)
        {
                tensor_kernel_t kernel(rows, cols);
                math::for_each(kernel, [&] (matrix_t& m)
                {
                        m.resize(o, i);
                });

                return kernel;
        }

        //-------------------------------------------------------------------------------------------------
}

