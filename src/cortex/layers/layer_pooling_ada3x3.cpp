#include "math/clamp.hpp"
#include "math/random.hpp"
#include "cortex/logger.h"
#include "tensor/numeric.hpp"
#include "text/from_params.hpp"
#include "tensor/serialize.hpp"
#include "layer_pooling_ada3x3.h"

namespace nano
{
        pooling_ada3x3_layer_t::pooling_ada3x3_layer_t(const string_t& parameters) :
                layer_t(parameters)
        {
        }

        tensor_size_t pooling_ada3x3_layer_t::resize(const tensor3d_t& tensor)
        {
                const auto idims = tensor.size<0>();
                const auto irows = tensor.size<1>();
                const auto icols = tensor.size<2>();

                const auto odims = idims;
                const auto orows = irows / 2;
                const auto ocols = icols / 2;

                m_idata.resize(idims, irows, icols);
                m_odata.resize(odims, orows, ocols);
                m_wdata.resize(odims, 3, 3);

                return psize();
        }

        void pooling_ada3x3_layer_t::zero_params()
        {
                tensor::set_zero(m_wdata);
        }

        void pooling_ada3x3_layer_t::random_params(scalar_t min, scalar_t max)
        {
                tensor::set_random(nano::random_t<scalar_t>(min, max), m_wdata);
        }

        scalar_t* pooling_ada3x3_layer_t::save_params(scalar_t* params) const
        {
                return tensor::to_array(params, m_wdata);
        }

        const scalar_t* pooling_ada3x3_layer_t::load_params(const scalar_t* params)
        {
                return tensor::from_array(params, m_wdata);
        }

        const tensor3d_t& pooling_ada3x3_layer_t::output(const tensor3d_t& input)
        {
                assert(idims() == input.size<0>());
                assert(irows() <= input.size<1>());
                assert(icols() <= input.size<2>());

                m_idata = input;

                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        const auto idata = m_idata.matrix(o);
                        const auto wdata = m_wdata.matrix(o);

                        auto odata = m_odata.matrix(o);

                        for (tensor_size_t r = 1; r < idata.rows(); r += 2)
                        {
                                for (tensor_size_t c = 1; c < idata.cols(); c += 2)
                                {
                                        const auto c0 = c - 1, c1 = c, c2 = std::min(c + 1, idata.cols() - 1);
                                        const auto r0 = r - 1, r1 = r, r2 = std::min(r + 1, idata.rows() - 1);

                                        odata(r / 2, c / 2) =
                                        idata(r0, c0) * wdata(0, 0) +
                                        idata(r0, c1) * wdata(0, 1) +
                                        idata(r0, c2) * wdata(0, 2) +
                                        idata(r1, c0) * wdata(1, 0) +
                                        idata(r1, c1) * wdata(1, 1) +
                                        idata(r1, c2) * wdata(1, 2) +
                                        idata(r2, c0) * wdata(2, 0) +
                                        idata(r2, c1) * wdata(2, 1) +
                                        idata(r2, c2) * wdata(2, 2);
                                }
                        }
                }

                return m_odata;
        }

        const tensor3d_t& pooling_ada3x3_layer_t::ginput(const tensor3d_t& output)
        {
                assert(odims() == output.size<0>());
                assert(orows() == output.size<1>());
                assert(ocols() == output.size<2>());

                m_odata = output;

                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        const auto odata = m_odata.matrix(o);
                        const auto wdata = m_wdata.matrix(o);

                        auto idata = m_idata.matrix(o);
                        idata.setZero();

                        for (tensor_size_t r = 1; r < idata.rows(); r += 2)
                        {
                                for (tensor_size_t c = 1; c < idata.cols(); c += 2)
                                {
                                        const auto c0 = c - 1, c1 = c, c2 = std::min(c + 1, idata.cols() - 1);
                                        const auto r0 = r - 1, r1 = r, r2 = std::min(r + 1, idata.rows() - 1);

                                        idata(r0, c0) += odata(r / 2, c / 2) * wdata(0, 0);
                                        idata(r0, c1) += odata(r / 2, c / 2) * wdata(0, 1);
                                        idata(r0, c2) += odata(r / 2, c / 2) * wdata(0, 2);

                                        idata(r1, c0) += odata(r / 2, c / 2) * wdata(1, 0);
                                        idata(r1, c1) += odata(r / 2, c / 2) * wdata(1, 1);
                                        idata(r1, c2) += odata(r / 2, c / 2) * wdata(1, 2);

                                        idata(r2, c0) += odata(r / 2, c / 2) * wdata(2, 0);
                                        idata(r2, c1) += odata(r / 2, c / 2) * wdata(2, 1);
                                        idata(r2, c2) += odata(r / 2, c / 2) * wdata(2, 2);
                                }
                        }
                }

                return m_idata;
        }

        void pooling_ada3x3_layer_t::gparam(const tensor3d_t& output, scalar_t* gradient)
        {
                assert(odims() == output.size<0>());
                assert(orows() == output.size<1>());
                assert(ocols() == output.size<2>());

                m_odata = output;

                auto gwdata = tensor::map_tensor(gradient, odims(), 3, 3);

                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        const auto idata = m_idata.matrix(o);
                        const auto odata = m_odata.matrix(o);

                        auto wdata = gwdata.matrix(o);
                        wdata.setZero();

                        for (tensor_size_t r = 1; r < idata.rows(); r += 2)
                        {
                                for (tensor_size_t c = 1; c < idata.cols(); c += 2)
                                {
                                        const auto c0 = c - 1, c1 = c, c2 = std::min(c + 1, idata.cols() - 1);
                                        const auto r0 = r - 1, r1 = r, r2 = std::min(r + 1, idata.rows() - 1);

                                        wdata(0, 0) += odata(r / 2, c / 2) * idata(r0, c0);
                                        wdata(0, 1) += odata(r / 2, c / 2) * idata(r0, c1);
                                        wdata(0, 2) += odata(r / 2, c / 2) * idata(r0, c2);

                                        wdata(1, 0) += odata(r / 2, c / 2) * idata(r1, c0);
                                        wdata(1, 1) += odata(r / 2, c / 2) * idata(r1, c1);
                                        wdata(1, 2) += odata(r / 2, c / 2) * idata(r1, c2);

                                        wdata(2, 0) += odata(r / 2, c / 2) * idata(r2, c0);
                                        wdata(2, 1) += odata(r / 2, c / 2) * idata(r2, c1);
                                        wdata(2, 2) += odata(r / 2, c / 2) * idata(r2, c2);
                                }
                        }
                }
        }
}


