#pragma once

#include "conv2d_dot.hpp"
#include "conv2d_mad.hpp"

namespace tensor
{
        struct conv2d_nx3_t
        {
                template
                <
                        typename tmatrixi,
                        typename tmatrixk,
                        typename tmatrixo
                >
                void operator()(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo&& odata) const
                {
                        conv2d_assert(idata, kdata, odata);
                        assert(kdata.cols() == 3);

                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto krows = kdata.rows();

                        for (auto r = 0; r < orows; r ++)
                        {
                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        odata.row(r) +=
                                        idata.row(r + kr).segment(0, ocols) * kdata(kr, 0) +
                                        idata.row(r + kr).segment(1, ocols) * kdata(kr, 1) +
                                        idata.row(r + kr).segment(2, ocols) * kdata(kr, 2);
                                }
                        }
                }
        };

        struct conv2d_nx5_t
        {
                template
                <
                        typename tmatrixi,
                        typename tmatrixk,
                        typename tmatrixo
                >
                void operator()(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo&& odata) const
                {
                        conv2d_assert(idata, kdata, odata);
                        assert(kdata.cols() == 5);

                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto krows = kdata.rows();

                        for (auto r = 0; r < orows; r ++)
                        {
                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        odata.row(r) +=
                                        idata.row(r + kr).segment(0, ocols) * kdata(kr, 0) +
                                        idata.row(r + kr).segment(1, ocols) * kdata(kr, 1) +
                                        idata.row(r + kr).segment(2, ocols) * kdata(kr, 2) +
                                        idata.row(r + kr).segment(3, ocols) * kdata(kr, 3) +
                                        idata.row(r + kr).segment(4, ocols) * kdata(kr, 4);
                                }
                        }
                }
        };


        ///
        /// \brief 2D convolution: odata += idata @ kdata (decode at runtime the best operator to use)
        ///
        struct conv2d_fix_t
        {
                template
                <
                        typename tmatrixi,
                        typename tmatrixk,
                        typename tmatrixo
                >
                void operator()(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo&& odata) const
                {
                        conv2d_assert(idata, kdata, odata);

                        if (kdata.rows() == 1 && kdata.cols() == 1)
                        {
                                odata.noalias() += idata * kdata(0, 0);
                        }
                        else if (odata.rows() == 1 && odata.cols() == 1)
                        {
                                odata(0, 0) += (idata.array() * kdata.array()).sum();
                        }
                        else
                        {
                                switch (kdata.cols())
                                {
                                case 3:         conv2d_nx3_t()(idata, kdata, odata); break;
                                case 5:         conv2d_nx5_t()(idata, kdata, odata); break;
                                default:        conv2d_dot_dyn_t()(idata, kdata, odata); break;
                                }
                        }
                }
        };
}

