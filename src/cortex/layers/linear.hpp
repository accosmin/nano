#pragma once

namespace cortex
{
        namespace linear
        {
                ///
                /// \brief linear output
                ///
                template
                <
                        typename ttensori,
                        typename ttensorw,
                        typename ttensorb,
                        typename ttensoro
                >
                void output(const ttensori& idata, const ttensorw& wdata, const ttensorb& bdata, ttensoro&& odata)
                {
                        odata.vector() = bdata + wdata * idata.vector();
                }

                ///
                /// \brief gradient wrt the input
                ///
                template
                <
                        typename ttensori,
                        typename ttensorw,
                        typename ttensorb,
                        typename ttensoro
                >
                void ginput(ttensori&& gidata, const ttensorw& wdata, const ttensorb&, const ttensoro& odata)
                {
                        gidata.vector() = wdata.transpose() * odata.vector();
                }

                ///
                /// \brief gradient wrt the parameters
                ///
                template
                <
                        typename ttensori,
                        typename ttensorw,
                        typename ttensorb,
                        typename ttensoro
                >
                void gparam(const ttensori& idata, ttensorw&& gwdata, ttensorb&& gbdata, const ttensoro& odata)
                {
                        gbdata = odata.vector();
                        gwdata = odata.vector() * idata.vector().transpose();
                }
        }
}

