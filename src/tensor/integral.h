#pragma once

#include "tensor.h"

namespace nano
{
        ///
        /// \brief compute the integral of a tensor of arbitrary rank.
        ///
        template <typename tstoragei, typename tstorageo, std::size_t trank>
        void integral(const tensor_t<tstoragei, trank>& itensor, tensor_t<tstorageo, trank>& otensor)
        {
                assert(itensor.dims() == otensor.dims());

                if (itensor.size() == 0)
                {
                        return;
                }

                const auto& dims = itensor.dims();

                // todo: generic N-dimensional integral

                if (dims.size() == 1)
                {
                        otensor(0) = itensor(0);
                        for (tensor_size_t i0 = 1, size0 = itensor.template size<0>(); i0 < size0; ++ i0)
                        {
                                otensor(i0) = otensor(i0 - 1) + itensor(i0);
                        }
                }
                /*else if (dims.size() == 2)
                {
                        otensor(0) = itensor(0);
                        for (tensor_size_t i0 = 1, size0 = itensor.template size<0>(); i0 < size0; ++ i0)
                        {
                                otensor(i0, 0) = otensor(i0 - 1, 0) + itensor(i0, 0);
                                for (tensor_size_t i1 = 1, size1 = itensor.template size<0>(); i1 < size1; ++ i1)
                                {
                                        otensor(i0, i1) = otensor(i0 - 1, i1) + itensor(i0, i1);
                                }
                        }
                }*/
        }
}
