#include "layer_linear.h"
#include "common/math.hpp"
#include "common/random.hpp"
#include "tensor/serialize.hpp"

namespace ncv
{
        linear_layer_t::linear_layer_t(const string_t& parameters)
                :       layer_t(parameters)
        {
        }

        size_t linear_layer_t::resize(const tensor_t& tensor)
        {
                const size_t idims = tensor.size();
                const size_t odims = math::clamp(text::from_params<size_t>(configuration(), "dims", 10),
                                                 size_t(1), size_t(4096));

                // resize buffers
                m_idata.resize(tensor.dims(), tensor.rows(), tensor.cols());
                m_odata.resize(odims, 1, 1);

                m_wdata.resize(1, odims, idims);
                m_bdata.resize(odims, 1, 1);

                return psize();
        }

        void linear_layer_t::zero_params()
        {
                m_wdata.zero();
                m_bdata.zero();
        }

        void linear_layer_t::random_params(scalar_t min, scalar_t max)
        {
                m_wdata.random(random_t<scalar_t>(min, max));
                m_bdata.random(random_t<scalar_t>(min, max));
        }

        scalar_t* linear_layer_t::save_params(scalar_t* params) const
        {
                params = tensor::save(m_wdata, params);
                params = tensor::save(m_bdata, params);
                return params;
        }

        const scalar_t* linear_layer_t::load_params(const scalar_t* params)
        {
                params = tensor::load(m_wdata, params);
                params = tensor::load(m_bdata, params);
                return params;
        }

        boost::archive::binary_oarchive& linear_layer_t::save(boost::archive::binary_oarchive& oa) const
        {
                return oa << m_wdata << m_bdata;
        }

        boost::archive::binary_iarchive& linear_layer_t::load(boost::archive::binary_iarchive& ia)
        {
                return ia >> m_wdata >> m_bdata;
        }

        const tensor_t& linear_layer_t::output(const tensor_t& input)
        {
                assert(idims() == input.dims());
                assert(irows() == input.rows());
                assert(icols() == input.cols());

                m_idata.copy_from(input);
                
                tensor::make_vector(m_odata.data(), osize()) = 
                        tensor::make_vector(m_bdata.data(), osize()) +
                        tensor::make_matrix(m_wdata.data(), osize(), isize()) *
                        tensor::make_vector(m_idata.data(), isize());;

                return m_odata;
        }

        const tensor_t& linear_layer_t::igrad(const tensor_t& output)
        {
                assert(output.dims() == odims());
                assert(output.rows() == orows());
                assert(output.cols() == ocols());

                m_odata.copy_from(output);
                
                tensor::make_vector(m_idata.data(), isize()).noalias() =
                        tensor::make_matrix(m_wdata.data(), osize(), isize()).transpose() *
                        tensor::make_vector(m_odata.data(), osize());

                return m_idata;
        }

        void linear_layer_t::pgrad(const tensor_t& output, scalar_t* gradient)
        {
                assert(output.dims() == odims());
                assert(output.rows() == orows());
                assert(output.cols() == ocols());

                m_odata.copy_from(output);
                                
                tensor::make_matrix(gradient, osize(), isize()).noalias() =
                        tensor::make_vector(m_odata.data(), osize()) *
                        tensor::make_vector(m_idata.data(), isize()).transpose();                
                        
                tensor::make_vector(gradient + osize() * isize(), osize()).noalias() = 
                        tensor::make_vector(m_odata.data(), osize());                
        }
}

