#pragma once

#include "nanocv/tensor/vector.hpp"
#include "nanocv/tensor/matrix.hpp"

namespace ncv
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
                void output(const ttensori& idata, const ttensorw& wdata, const ttensorb& bdata, ttensoro& odata)
                {
                        odata.vector() = bdata.vector() + wdata.matrix(0) * idata.vector();
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
                void ginput(ttensori& gidata, const ttensorw& wdata, const ttensorb&, const ttensoro& odata)
                {
                        gidata.vector() = wdata.matrix(0).transpose() * odata.vector();
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
                void gparam(const ttensori& idata, ttensorw& gwdata, ttensorb& gbdata, const ttensoro& odata)
                {
                        gbdata.vector() = odata.vector();
                        gwdata.matrix(0) = odata.vector() * idata.vector().transpose();
                }
        }
}

