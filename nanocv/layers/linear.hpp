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
                        typename ttensor
                >
                void output(const ttensor& idata, const ttensor& wdata, const ttensor& bdata, ttensor& odata)
                {
                        odata.vector() = bdata.vector() + wdata.matrix(0) * idata.vector();
                }

                ///
                /// \brief gradient wrt the input
                ///
                template
                <
                        typename ttensor
                >
                void ginput(ttensor& gidata, const ttensor& wdata, const ttensor&, const ttensor& odata)
                {
                        gidata.vector() = wdata.matrix(0).transpose() * odata.vector();
                }

                ///
                /// \brief gradient wrt the parameters
                ///
                template
                <
                        typename ttensor
                >
                void gparam(const ttensor& idata, ttensor& gwdata, ttensor& gbdata, const ttensor& odata)
                {
                        gbdata.vector() = odata.vector();
                        gwdata.matrix(0) = odata.vector() * idata.vector().transpose();
                }
        }
}

