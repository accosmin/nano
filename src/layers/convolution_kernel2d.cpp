#include "math/clamp.hpp"
#include "math/random.hpp"
#include "logger.h"
#include "tensor/numeric.hpp"
#include "text/to_string.hpp"
#include "text/from_params.hpp"
#include "tensor/serialize.hpp"
#include "tensor/conv2d_dyn.hpp"
#include "tensor/corr2d_dyn.hpp"
#include "convolution_kernel2d.h"

namespace nano
{
        template <typename top>
        void conv_loop(const tensor_size_t odims, const tensor_size_t idims, const tensor_size_t kconn, const top& op)
        {
                for (tensor_size_t o = 0; o < odims; ++ o)
                {
                        for (tensor_size_t i = (o % kconn), ik = 0; i < idims; ++ ik, i += kconn)
                        {
                                op(i, ik, o);
                        }
                }
        }

        conv_layer_kernel2d_t::conv_layer_kernel2d_t(const string_t& parameters) :
                layer_t(parameters),
                m_kconn(1)
        {
        }

        tensor_size_t conv_layer_kernel2d_t::resize(const tensor3d_t& tensor)
        {
                const auto idims = tensor.size<0>();
                const auto irows = tensor.size<1>();
                const auto icols = tensor.size<2>();

                const auto odims = clamp(from_params<tensor_size_t>(configuration(), "dims", 16), 1, 256);
                const auto krows = clamp(from_params<tensor_size_t>(configuration(), "rows", 8), 1, 32);
                const auto kcols = clamp(from_params<tensor_size_t>(configuration(), "cols", 8), 1, 32);
                const auto kconn = clamp(from_params<tensor_size_t>(configuration(), "conn", 1), 1, 16);

                const auto orows = irows - krows + 1;
                const auto ocols = icols - kcols + 1;

                // check convolution size
                if (irows < krows || icols < kcols)
                {
                        log_error() << "convolution layer: invalid size (" << idims << "x" << irows << "x" << icols
                                    << ") -> (" << odims << "x" << krows << "x" << kcols << ")!";
                        throw std::invalid_argument("invalid configuration for the convolution layer");
                }

                // check input connectivity factor
                if ((idims % kconn) || (odims % kconn))
                {
                        log_error() << "convolution layer: invalid input connectivity factor!";
                        throw std::invalid_argument("invalid configuration for the convolution layer");
                }

                // resize buffers
                m_idata.resize(idims, irows, icols);
                m_odata.resize(odims, orows, ocols);
                m_kdata.resize(odims, idims / kconn, krows, kcols);
                m_bdata.resize(odims, 1, 1);
                m_kconn = kconn;

                return psize();
        }

        void conv_layer_kernel2d_t::zero_params()
        {
                tensor::set_zero(m_kdata, m_bdata);
        }

        void conv_layer_kernel2d_t::random_params(scalar_t min, scalar_t max)
        {
                tensor::set_random(random_t<scalar_t>(min, max), m_kdata, m_bdata);
        }

        scalar_t* conv_layer_kernel2d_t::save_params(scalar_t* params) const
        {
                return tensor::to_array(params, m_kdata, m_bdata);
        }

        const scalar_t* conv_layer_kernel2d_t::load_params(const scalar_t* params)
        {
                return tensor::from_array(params, m_kdata, m_bdata);
        }

        const tensor3d_t& conv_layer_kernel2d_t::output(const tensor3d_t& input)
        {
                assert(idims() == input.size<0>());
                assert(irows() == input.size<1>());
                assert(icols() == input.size<2>());

                m_idata = input;

                // convolution
                m_odata.setZero();

                tensor::conv2d_dyn_t op;
                conv_loop(odims(), idims(), kconn(), [&] (const auto i, const auto ik, const auto o)
                {
                        op(m_idata.matrix(i), m_kdata.matrix(o, ik), m_odata.matrix(o));
                });

                // +bias
                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        m_odata.vector(o).array() += m_bdata(o);
                }

                return m_odata;
        }

        const tensor3d_t& conv_layer_kernel2d_t::ginput(const tensor3d_t& output)
        {
                assert(odims() == output.size<0>());
                assert(orows() == output.size<1>());
                assert(ocols() == output.size<2>());

                m_odata = output;

                m_idata.setZero();

                tensor::corr2d_dyn_t op;
                conv_loop(odims(), idims(), kconn(), [&] (const auto i, const auto ik, const auto o)
                {
                        op(m_odata.matrix(o), m_kdata.matrix(o, ik), m_idata.matrix(i));
                });

                return m_idata;
        }

        void conv_layer_kernel2d_t::gparam(const tensor3d_t& output, scalar_t* gradient)
        {
                assert(odims() == output.size<0>());
                assert(orows() == output.size<1>());
                assert(ocols() == output.size<2>());

                m_odata = output;

                // wrt convolution
                auto gkdata = tensor::map_tensor(gradient, m_kdata.size<0>(), m_kdata.size<1>(), krows(), kcols());

                gkdata.setZero();

                tensor::conv2d_dyn_t op;
                conv_loop(odims(), idims(), kconn(), [&] (const auto i, const auto ik, const auto o)
                {
                        op(m_idata.matrix(i), m_odata.matrix(o), gkdata.matrix(o, ik));
                });

                // wrt bias
                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        gradient[m_kdata.size() + o] = m_odata.vector(o).sum();
                }
        }
}


