#pragma once

#include "tensor.h"

namespace nano
{
        ///
        /// \brief utilities to down-sample by 2 using 3x3 overlapping regions.
        ///
        namespace pooling
        {
                template <typename tiplane, typename toplane, typename toperator>
                void output(const tiplane& iplane, toplane&& oplane, const toperator& op)
                {
                        for (tensor_size_t r = 1; r < iplane.rows(); r += 2)
                        {
                                for (tensor_size_t c = 1; c < iplane.cols(); c += 2)
                                {
                                        const tensor_size_t c0 = c - 1, c1 = c, c2 = std::min(c + 1, iplane.cols() - 1);
                                        const tensor_size_t r0 = r - 1, r1 = r, r2 = std::min(r + 1, iplane.rows() - 1);

                                        oplane(r / 2, c / 2) = op(
                                                iplane(r0, c0), iplane(r0, c1), iplane(r0, c2),
                                                iplane(r1, c0), iplane(r1, c1), iplane(r1, c2),
                                                iplane(r2, c0), iplane(r2, c1), iplane(r2, c2));
                                }
                        }
                }

                template <typename tiplane, typename toplane, typename toperator>
                void ginput(tiplane&& iplane, const toplane& oplane, const toperator& op)
                {
                        iplane.setZero();

                        for (tensor_size_t r = 1; r < iplane.rows(); r += 2)
                        {
                                for (tensor_size_t c = 1; c < iplane.cols(); c += 2)
                                {
                                        const tensor_size_t c0 = c - 1, c1 = c, c2 = std::min(c + 1, iplane.cols() - 1);
                                        const tensor_size_t r0 = r - 1, r1 = r, r2 = std::min(r + 1, iplane.rows() - 1);

                                        op(     oplane(r / 2, c / 2),
                                                iplane(r0, c0), iplane(r0, c1), iplane(r0, c2),
                                                iplane(r1, c0), iplane(r1, c1), iplane(r1, c2),
                                                iplane(r2, c0), iplane(r2, c1), iplane(r2, c2));
                                }
                        }
                }

                template <typename tiplane, typename toplane, typename toperator>
                void gparam(const tiplane& iplane, const toplane& oplane, const toperator& op)
                {
                        for (tensor_size_t r = 1; r < iplane.rows(); r += 2)
                        {
                                for (tensor_size_t c = 1; c < iplane.cols(); c += 2)
                                {
                                        const tensor_size_t c0 = c - 1, c1 = c, c2 = std::min(c + 1, iplane.cols() - 1);
                                        const tensor_size_t r0 = r - 1, r1 = r, r2 = std::min(r + 1, iplane.rows() - 1);

                                        const auto ooo = oplane(r / 2, c / 2);

                                        op(     ooo * iplane(r0, c0), ooo * iplane(r0, c1), ooo * iplane(r0, c2),
                                                ooo * iplane(r1, c0), ooo * iplane(r1, c1), ooo * iplane(r1, c2),
                                                ooo * iplane(r2, c0), ooo * iplane(r2, c1), ooo * iplane(r2, c2));
                                }
                        }
                }
        }
}


