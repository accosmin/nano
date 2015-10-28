#include "layer_convolution.h"
#include "math/clamp.hpp"
#include "math/random.hpp"
#include "tensor/random.hpp"
#include "tensor/conv3d.hpp"
#include "text/to_string.hpp"
#include "cortex/util/logger.h"
#include "text/from_params.hpp"
#include "tensor/serialize.hpp"
#include "tensor/conv2d_dyn.hpp"
#include "tensor/corr2d_dyn.hpp"

namespace cortex
{
        conv_layer_t::conv_layer_t(const string_t& parameters)
                :       layer_t(parameters)
        {
        }

        conv_layer_t::~conv_layer_t()
        {
        }

        tensor_size_t conv_layer_t::resize(const tensor_t& tensor)
        {
                const auto idims = tensor.dims();
                const auto irows = tensor.rows();
                const auto icols = tensor.cols();

                const auto odims = math::clamp(text::from_params<tensor_size_t>(configuration(), "dims", 16), 1, 256);
                const auto krows = math::clamp(text::from_params<tensor_size_t>(configuration(), "rows", 8), 1, 32);
                const auto kcols = math::clamp(text::from_params<tensor_size_t>(configuration(), "cols", 8), 1, 32);

                // check convolution size
                if (irows < krows || icols < kcols)
                {
                        const string_t message =
                                "invalid size (" + text::to_string(idims) + "x" + text::to_string(irows) +
                                 "x" + text::to_string(icols) + ") -> (" + text::to_string(odims) + "x" +
                                 text::to_string(krows) + "x" + text::to_string(kcols) + ")";

                        log_error() << "convolution layer: " << message;
                        throw std::runtime_error("convolution layer: " + message);
                }

                const tensor_size_t orows = irows - krows + 1;
                const tensor_size_t ocols = icols - kcols + 1;

                // resize buffers
                m_idata.resize(idims, irows, icols);
                m_odata.resize(odims, orows, ocols);
                m_kdata.resize(idims * odims, krows, kcols);
                m_bdata.resize(odims, 1, 1);

                return psize();
        }

        void conv_layer_t::zero_params()
        {
                m_kdata.setZero();
                m_bdata.setZero();
        }

        void conv_layer_t::random_params(scalar_t min, scalar_t max)
        {
                tensor::set_random(m_kdata, math::random_t<scalar_t>(min, max));
                tensor::set_random(m_bdata, math::random_t<scalar_t>(min, max));
        }

        scalar_t* conv_layer_t::save_params(scalar_t* params) const
        {
                params = tensor::save(m_kdata, params);
                params = tensor::save(m_bdata, params);

                return params;
        }

        const scalar_t* conv_layer_t::load_params(const scalar_t* params)
        {
                params = tensor::load(m_kdata, params);
                params = tensor::load(m_bdata, params);

                return params;
        }

        tensor_size_t conv_layer_t::psize() const
        {
                return m_kdata.size() + m_bdata.size();
        }

        const tensor_t& conv_layer_t::output(const tensor_t& input)
        {
                assert(idims() == input.dims());
                assert(irows() == input.rows());
                assert(icols() == input.cols());

                m_idata = input;

                // convolution
                tensor::conv3d_output(tensor::conv2d_dyn_t(), m_idata, m_kdata, m_odata);

                // +bias
                for (tensor_size_t o = 0; o < odims(); o ++)
                {
                        m_odata.vector(o).array() += m_bdata(o);
                }

                return m_odata;
        }        

        const tensor_t& conv_layer_t::ginput(const tensor_t& output)
        {
                assert(odims() == output.dims());
                assert(orows() == output.rows());
                assert(ocols() == output.cols());

                m_odata = output;

                tensor::conv3d_ginput(tensor::corr2d_dyn_t(), m_idata, m_kdata, m_odata);

                return m_idata;
        }

        void conv_layer_t::gparam(const tensor_t& output, scalar_t* gradient)
        {
                assert(odims() == output.dims());
                assert(orows() == output.rows());
                assert(ocols() == output.cols());

                m_odata = output;
                
                // wrt convolution
                auto kdata = tensor::map_tensor(gradient, m_kdata.dims(), m_kdata.rows(), m_kdata.cols());
                tensor::conv3d_gparam(tensor::conv2d_dyn_t(), m_idata, kdata, m_odata);

                // wrt bias
                for (tensor_size_t o = 0; o < odims(); o ++)
                {
                        gradient[m_kdata.size() + o] = m_odata.vector(o).sum();
                }
        }
}


