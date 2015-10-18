#include "layer_convolution.h"
#include "math/clamp.hpp"
#include "math/random.hpp"
#include "cortex/logger.h"
#include "tensor/random.hpp"
#include "text/to_string.hpp"
#include "text/from_params.hpp"
#include "tensor/serialize.hpp"

namespace cortex
{
        conv_layer_t::conv_layer_t(const string_t& parameters)
                :       layer_t(parameters)
        {
        }

        conv_layer_t::~conv_layer_t()
        {
        }

        size_t conv_layer_t::resize(const tensor_t& tensor)
        {
                const size_t idims = tensor.dims();
                const size_t irows = tensor.rows();
                const size_t icols = tensor.cols();

                const size_t odims = math::clamp(text::from_params<size_t>(configuration(), "dims", 16), 1, 256);
                const size_t krows = math::clamp(text::from_params<size_t>(configuration(), "rows", 8), 1, 32);
                const size_t kcols = math::clamp(text::from_params<size_t>(configuration(), "cols", 8), 1, 32);

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

                const size_t orows = irows - krows + 1;
                const size_t ocols = icols - kcols + 1;

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

                params_changed();
        }

        void conv_layer_t::random_params(scalar_t min, scalar_t max)
        {
                tensor::set_random(m_kdata, math::random_t<scalar_t>(min, max));
                tensor::set_random(m_bdata, math::random_t<scalar_t>(min, max));

                params_changed();
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

                params_changed();

                return params;
        }

        size_t conv_layer_t::psize() const
        {
                return m_kdata.size() + m_bdata.size();
        }

        void conv_layer_t::params_changed()
        {
                m_kconv.reset(m_kdata, idims(), odims());
        }

        const tensor_t& conv_layer_t::output(const tensor_t& input)
        {
                assert(idims() == static_cast<size_t>(input.dims()));
                assert(irows() == static_cast<size_t>(input.rows()));
                assert(icols() == static_cast<size_t>(input.cols()));

                m_idata = input;

                // convolution
                m_kconv.output(m_idata, m_odata);

                // +bias
                for (size_t o = 0; o < odims(); o ++)
                {
                        m_odata.vector(o).array() += m_bdata(o);
                }

                return m_odata;
        }        

        const tensor_t& conv_layer_t::ginput(const tensor_t& output)
        {
                assert(odims() == static_cast<size_t>(output.dims()));
                assert(orows() == static_cast<size_t>(output.rows()));
                assert(ocols() == static_cast<size_t>(output.cols()));

                m_odata = output;

                m_kconv.ginput(m_idata, m_odata);

                return m_idata;
        }

        void conv_layer_t::gparam(const tensor_t& output, scalar_t* gradient)
        {
                assert(odims() == static_cast<size_t>(output.dims()));
                assert(orows() == static_cast<size_t>(output.rows()));
                assert(ocols() == static_cast<size_t>(output.cols()));

                m_odata = output;
                
                // wrt convolution
                auto kdata = tensor::map_tensor(gradient, m_kdata.dims(), m_kdata.rows(), m_kdata.cols());
                m_kconv.gparam(m_idata, kdata, m_odata);

                // wrt bias
                for (size_t o = 0; o < odims(); o ++)
                {
                        gradient[m_kdata.size() + o] = m_odata.vector(o).sum();
                }
        }
}


