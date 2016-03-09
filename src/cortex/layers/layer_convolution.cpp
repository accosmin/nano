#include "layer_convolution.h"
#include "math/clamp.hpp"
#include "math/random.hpp"
#include "tensor/random.hpp"
#include "text/to_string.hpp"
#include "cortex/util/logger.h"
#include "text/from_params.hpp"
#include "tensor/serialize.hpp"
#include "tensor/conv2d_dyn.hpp"
#include "tensor/corr2d_dyn.hpp"

namespace zob
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

                const auto odims = zob::clamp(zob::from_params<tensor_size_t>(configuration(), "dims", 16), 1, 256);
                const auto krows = zob::clamp(zob::from_params<tensor_size_t>(configuration(), "rows", 8), 1, 32);
                const auto kcols = zob::clamp(zob::from_params<tensor_size_t>(configuration(), "cols", 8), 1, 32);

                const auto kdims = idims * odims;
                const auto orows = irows - krows + 1;
                const auto ocols = icols - kcols + 1;

                // check convolution size
                if (irows < krows || icols < kcols)
                {
                        log_error() << "convolution layer: invalid size (" << idims << "x" << irows << "x" << icols
                                    << ") -> (" << odims << "x" << krows << "x" << kcols << ")!";
                        throw std::invalid_argument("invalid configuration for the convolution layer");
                }

                // resize buffers
                m_idata.resize(idims, irows, icols);
                m_odata.resize(odims, orows, ocols);
                m_kdata.resize(kdims, krows, kcols);
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
                tensor::set_random(m_kdata, zob::random_t<scalar_t>(min, max));
                tensor::set_random(m_bdata, zob::random_t<scalar_t>(min, max));
        }

        scalar_t* conv_layer_t::save_params(scalar_t* params) const
        {
                params = tensor::to_array(m_kdata, params);
                params = tensor::to_array(m_bdata, params);
                return params;
        }

        const scalar_t* conv_layer_t::load_params(const scalar_t* params)
        {
                params = tensor::from_array(m_kdata, params);
                params = tensor::from_array(m_bdata, params);
                return params;
        }

        const tensor_t& conv_layer_t::output(const tensor_t& input)
        {
                assert(idims() == input.dims());
                assert(irows() == input.rows());
                assert(icols() == input.cols());

                m_idata = input;

                // convolution
                m_odata.setZero();

                tensor::conv2d_dyn_t op;
                for (tensor_size_t i = 0, k = 0; i < idims(); ++ i)
                {
                        for (tensor_size_t o = 0; o < odims(); ++ o, ++ k)
                        {
                                op(m_idata.matrix(i), m_kdata.matrix(k), m_odata.matrix(o));
                        }
                }

                // +bias
                for (tensor_size_t o = 0; o < odims(); ++ o)
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

                m_idata.setZero();

                tensor::corr2d_dyn_t op;
                for (tensor_size_t i = 0, k = 0; i < idims(); ++ i)
                {
                        for (tensor_size_t o = 0; o < odims(); ++ o, ++ k)
                        {
                                op(m_odata.matrix(o), m_kdata.matrix(k), m_idata.matrix(i));
                        }
                }

                return m_idata;
        }

        void conv_layer_t::gparam(const tensor_t& output, scalar_t* gradient)
        {
                assert(odims() == output.dims());
                assert(orows() == output.rows());
                assert(ocols() == output.cols());

                m_odata = output;
                
                // wrt convolution
                auto gkdata = tensor::map_tensor(gradient, kdims(), krows(), kcols());
                gkdata.setZero();

                tensor::conv2d_dyn_t op;
                for (tensor_size_t i = 0, k = 0; i < idims(); ++ i)
                {
                        for (tensor_size_t o = 0; o < odims(); ++ o, ++ k)
                        {
                                op(m_idata.matrix(i), m_odata.matrix(o), gkdata.matrix(k));
                        }
                }

                // wrt bias
                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        gradient[m_kdata.size() + o] = m_odata.vector(o).sum();
                }
        }
}


