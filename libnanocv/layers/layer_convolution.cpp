#include "layer_convolution.h"
#include "libnanocv/util/logger.h"
#include "libnanocv/util/clamp.hpp"
#include "libnanocv/util/random.hpp"
#include "libnanocv/util/sampling.hpp"
#include "libnanocv/tensor/serialize.hpp"
#include "convolution.hpp"

namespace ncv
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
                m_kdata.resize(odims * idims, krows, kcols);
                m_bdata.resize(odims, 1, 1);

                return psize();
        }

        void conv_layer_t::zero_params()
        {
                m_kdata.zero();
                m_bdata.zero();
        }

        void conv_layer_t::random_params(scalar_t min, scalar_t max)
        {
                m_kdata.random(random_t<scalar_t>(min, max));
                m_bdata.random(random_t<scalar_t>(min, max));
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

        boost::archive::binary_oarchive& conv_layer_t::save(boost::archive::binary_oarchive& oa) const
        {
                return oa << m_kdata << m_bdata;
        }

        boost::archive::binary_iarchive& conv_layer_t::load(boost::archive::binary_iarchive& ia)
        {
                return ia >> m_kdata >> m_bdata;
        }

        size_t conv_layer_t::psize() const
        {
                return m_kdata.size() + m_bdata.size();
        }

        const tensor_t& conv_layer_t::output(const tensor_t& input)
        {
                assert(idims() == input.dims());
                assert(irows() == input.rows());
                assert(icols() == input.cols());

                m_idata.copy_from(input);
                
                // convolution
                convolution::output(
                        m_idata.data(), idims(),
                        m_kdata.data(), krows(), kcols(),
                        m_odata.data(), odims(), orows(), ocols());

                // +bias
                for (size_t o = 0; o < odims(); o ++)
                {
                        m_odata.plane_vector(o).array() += m_bdata.data(o);
                }

                return m_odata;
        }        

        const tensor_t& conv_layer_t::ginput(const tensor_t& output)
        {
                assert(odims() == output.dims());
                assert(orows() == output.rows());
                assert(ocols() == output.cols());

                m_odata.copy_from(output);
                
                convolution::ginput(
                        m_idata.data(), idims(),
                        m_kdata.data(), krows(), kcols(),
                        m_odata.data(), odims(), orows(), ocols());

                return m_idata;
        }

        void conv_layer_t::gparam(const tensor_t& output, scalar_t* gradient)
        {
                assert(odims() == output.dims());
                assert(orows() == output.rows());
                assert(ocols() == output.cols());

                m_odata.copy_from(output);
                
                // wrt convolution
                convolution::gparam(
                        m_idata.data(), idims(),
                        gradient, krows(), kcols(),
                        m_odata.data(), odims(), orows(), ocols());

                // wrt bias
                for (size_t o = 0; o < odims(); o ++)
                {
                        gradient[m_kdata.size() + o] = m_odata.plane_vector(o).sum();
                }
        }
}


