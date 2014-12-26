#include "layer_convolution.h"
#include "io/logger.h"
#include "common/math.hpp"
#include "common/random.hpp"
#include "common/conv2d.hpp"
#include "common/corr2d.hpp"
#include "common/sampling.hpp"
#include "tensor/serialize.hpp"

namespace ncv
{
        static bool is_masked(scalar_t value)
        {
                return value > 0.5;
        }

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
                m_mdata.resize(odims, idims);

                make_mask();

                return psize();
        }

        void conv_layer_t::zero_params()
        {
                m_kdata.zero();

                make_mask();
        }

        void conv_layer_t::random_params(scalar_t min, scalar_t max)
        {
                m_kdata.random(random_t<scalar_t>(min, max));

                make_mask();
        }

        void conv_layer_t::make_mask()
        {
                const size_t mask = math::clamp(text::from_params<size_t>(configuration(), "mask", 100), 1, 100);

                if (mask >= 100)
                {
                        // full connection
                        m_mdata.setOnes();
                }

                else
                {
                        // connect with the given probability
                        m_mdata.setZero();
                        for (size_t o = 0; o < odims(); o ++)
                        {
                                const indices_t indices = uniform_indices(idims(), std::max(size_t(1), idims() * mask / 100));
                                for (size_t i : indices)
                                {
                                        m_mdata(o, i) = 1.0;
                                }
                        }
                }

//                for (size_t o = 0; o < odims(); o ++)
//                {
//                        string_t mask;
//                        for (size_t i = 0; i < idims(); i ++)
//                        {
//                                mask.append(is_masked(m_mdata(o, i)) ? "1" : "0");
//                        }

//                        log_info() << "mask [" << (o + 1) << "/" << odims() << "]: " << mask;
//                }
        }

        size_t conv_layer_t::mask_count() const
        {
                size_t count = 0;

                for (matrix_t::Index i = 0; i < m_mdata.size(); i ++)
                {
                        if (is_masked(m_mdata(i)))
                        {
                                count ++;
                        }
                }

                return count;
        }

        scalar_t* conv_layer_t::save_params(scalar_t* params) const
        {
                for (size_t o = 0, k = 0; o < odims(); o ++)
                {
                        for (size_t i = 0; i < idims(); i ++, k ++)
                        {
                                if (is_masked(m_mdata(o, i)))
                                {
                                        auto kmap = tensor::make_vector(m_kdata.plane_data(k), m_kdata.plane_size());
                                        params = tensor::save(kmap, params);
                                }
                        }
                }

                return params;
        }

        const scalar_t* conv_layer_t::load_params(const scalar_t* params)
        {
                for (size_t o = 0, k = 0; o < odims(); o ++)
                {
                        for (size_t i = 0; i < idims(); i ++, k ++)
                        {
                                if (is_masked(m_mdata(o, i)))
                                {
                                        auto kmap = tensor::make_vector(m_kdata.plane_data(k), m_kdata.plane_size());
                                        params = tensor::load(kmap, params);
                                }
                        }
                }

                return params;
        }

        boost::archive::binary_oarchive& conv_layer_t::save(boost::archive::binary_oarchive& oa) const
        {
                return oa << m_kdata << m_mdata;
        }

        boost::archive::binary_iarchive& conv_layer_t::load(boost::archive::binary_iarchive& ia)
        {
                return ia >> m_kdata >> m_mdata;
        }

        size_t conv_layer_t::psize() const
        {
                return static_cast<size_t>(m_mdata.sum()) * m_kdata.plane_size();
        }

        const tensor_t& conv_layer_t::output(const tensor_t& input)
        {
                assert(idims() == input.dims());
                assert(irows() == input.rows());
                assert(icols() == input.cols());

                m_idata.copy_from(input);
                
                for (size_t o = 0; o < odims(); o ++)
                {
                        auto omap = m_odata.plane_matrix(o);                        
                        
                        omap.setZero();                        
                        for (size_t i = 0; i < idims(); i ++)
                        {
                                auto imap = m_idata.plane_matrix(i);
                                auto kmap = m_kdata.plane_matrix(o * idims() + i);
                                
                                if (is_masked(m_mdata(o, i)))
                                {
                                        ncv::conv2d_dyn(imap, kmap, omap);
                                }
                        }
                }

                return m_odata;
        }        

        const tensor_t& conv_layer_t::ginput(const tensor_t& output)
        {
                assert(odims() == output.dims());
                assert(orows() == output.rows());
                assert(ocols() == output.cols());

                m_odata.copy_from(output);
                
                m_idata.zero();                
                for (size_t o = 0; o < odims(); o ++)
                {
                        auto omap = m_odata.plane_matrix(o);
                        
                        for (size_t i = 0; i < idims(); i ++)
                        {
                                auto gimap = m_idata.plane_matrix(i);
                                auto kmap = m_kdata.plane_matrix(o * idims() + i);
                                
                                if (is_masked(m_mdata(o, i)))
                                {
                                        ncv::corr2d_dyn(omap, kmap, gimap);
                                }
                        }
                }

                return m_idata;
        }

        void conv_layer_t::gparam(const tensor_t& output, scalar_t* gradient)
        {
                assert(odims() == output.dims());
                assert(orows() == output.rows());
                assert(ocols() == output.cols());

                m_odata.copy_from(output);
                
                const size_t ksize = krows() * kcols();
                
                for (size_t o = 0, k = 0; o < odims(); o ++)
                {
                        auto omap = m_odata.plane_matrix(o);
                        
                        for (size_t i = 0; i < idims(); i ++)
                        {
                                auto imap = m_idata.plane_matrix(i);
                                auto gkmap = tensor::make_matrix(gradient + k * ksize, krows(), kcols());
                                
                                if (is_masked(m_mdata(o, i)))
                                {
                                        gkmap.setZero();
                                        ncv::conv2d_dyn(imap, omap, gkmap);
                                        k ++;
                                }
                        }
                }
        }
}


