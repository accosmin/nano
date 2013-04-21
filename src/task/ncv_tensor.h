#ifndef NANOCV_TENSOR_H
#define NANOCV_TENSOR_H

#include "ncv_math.h"
#include "ncv_types.h"

namespace Eigen
{
        template<>
        struct NumTraits<ncv::vector_t> : NumTraits<ncv::scalar_t>
                        // permits to get the epsilon, dummy_precision, lowest, highest functions
        {
                typedef ncv::vector_t Real;
                typedef ncv::vector_t NonInteger;
                typedef ncv::vector_t Nested;
                enum
                {
                        IsComplex = 0,
                        IsInteger = 0,
                        IsSigned = 1,
                        RequireInitialization = 1,
                        ReadCost = 1,
                        AddCost = 3,
                        MulCost = 3
                };
        };

        template<>
        struct NumTraits<ncv::matrix_t> : NumTraits<ncv::scalar_t>
                        // permits to get the epsilon, dummy_precision, lowest, highest functions
        {
                typedef ncv::matrix_t Real;
                typedef ncv::matrix_t NonInteger;
                typedef ncv::matrix_t Nested;
                enum
                {
                        IsComplex = 0,
                        IsInteger = 0,
                        IsSigned = 1,
                        RequireInitialization = 1,
                        ReadCost = 1,
                        AddCost = 3,
                        MulCost = 3
                };
        };
}

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // 3D/4D tensors are used for storing and processing 2D data (e.g. image patch).
        // the 3D tensors map various vectorial attributes (e.g. colors channels),
        // while the 4D tensors transform some input vectorial attributes to
        // another set of attributes (of possible different number).
        ////////////////////////////////////////////////////////////////////////////////

        // 3D data tensor:
        //      (row, col) = Ix1 (e.g. I can be number of color/data channels)
        typedef matrix<vector_t>::matrix_t      tensor_data_t;

        // 4D kernel tensor:
        //      (row, col) = OxI (transforms I data channels to O data channels)
        typedef matrix<matrix_t>::matrix_t	tensor_kernel_t;

        namespace tensor
        {
                // create tensors
                tensor_data_t make(size_t rows, size_t cols, size_t i);
                tensor_kernel_t make(size_t rows, size_t cols, size_t o, size_t i);

                // initialize tensors (using a scalar generator)
                template
                <
                        typename ttensor,
                        typename tgenerator
                >
                void set_generator(ttensor& tensor, tgenerator gen)
                {
                        math::for_each(tensor, [&] (typename ttensor::Scalar& content)
                        {
                                math::for_each(content, [&] (scalar_t& v)
                                {
                                        v = gen();
                                });
                        });
                }

                // initialize tensors (using a scalar constant)
                template
                <
                        typename ttensor
                >
                void set_constant(ttensor& tensor, scalar_t val)
                {
                        set_generator(tensor, [&] ()
                        {
                                return val;
                        });
                }

                // transform tensors (for each scalar value)
                template
                <
                        typename ttensor,
                        typename toperator
                >
                void for_each_value(ttensor& tensor, toperator op)
                {
                        math::for_each(tensor, [&] (typename ttensor::Scalar& content)
                        {
                                math::for_each(content, [&] (scalar_t& v)
                                {
                                        v = op(v);
                                });
                        });
                }

                // transform tensors (for each 1D/2D element)
                template
                <
                        typename ttensor,
                        typename toperator
                >
                void for_each_element(ttensor& tensor, toperator op)
                {
                        math::for_each(tensor, [&] (typename ttensor::Scalar& content)
                        {
                                op(content);
                        });
                }

                // convolve the input tensor (Ix1) with the kernel (OxI) to produce the output tensor (Ox1)
                // NB: the ouput is assumed to be correctly resized!
                template
                <
                        typename ttensor_data
                >
                void conv(const tensor_data_t& input,
                          const tensor_kernel_t& kernel,
                          tensor_data_t& output)
                {
                        tensor::set_constant(output, 0.0);

                        const int rows = math::cast<int>(input.rows()), krows = math::cast<int>(kernel.rows());
                        const int cols = math::cast<int>(input.cols()), kcols = math::cast<int>(kernel.cols());
                        const int rmin = 0, rmax = rows - krows;
                        const int cmin = 0, cmax = cols - kcols;
                        const int ksize = krows * kcols;

                        for (int r = rmin; r < rmax; r ++)
                        {
                                for (int c = cmin; c < cmax; c ++)
                                {
                                        const auto& iblock = input.block(r, c, krows, kcols);

                                        for (int k = 0; k < ksize; k ++)
                                        {
                                                output(r, c).noalias() += kernel(k) * iblock(k);
                                        }
                                }
                        }
                }
        }
}

#endif // NANOCV_TENSOR_H
