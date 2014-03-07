#include "layer_linear.h"
#include "text.h"
#include "common/math.hpp"
#include "common/random.hpp"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                typename tscalar,
                typename tsize
        >
        static void _forward(
                const tscalar* idata, tsize isize,
                const tscalar* wdata,
                const tscalar* bdata,
                tscalar* odata, tsize osize)
        {
                for (tsize o = 0; o < osize; o ++)
                {
                        tscalar sum = bdata[o];
                        for (tsize i = 0; i < isize; i ++)
                        {
                                sum += wdata[o * isize + i] * idata[i];
                        }

                        odata[o] = sum;
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        static const string_t ocl_linear_source = R"xxx(

        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable

        __kernel void linear_forward(
                __global const double* restrict idata, int isize,
                __global const double* wdata,
                __constant const double* bdata,
                __global double* restrict odata)
        {
                const int osize = get_global_size(0);

                const int o = get_global_id(0);

                double sum = bdata[o];
                for (int i = 0; i < isize; i ++)
                {
                        sum += wdata[o * isize + i] * idata[i];
                }

                odata[o] = sum;
        }

        )xxx";

        /////////////////////////////////////////////////////////////////////////////////////////

        linear_layer_t::linear_layer_t(const string_t& parameters)
                :       layer_t(parameters, "fully-connected linear layer, parameters: dims=10[1,4096]")
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t linear_layer_t::resize(const tensor_t& tensor)
        {
                const size_t idims = tensor.size();
                const size_t odims = math::clamp(text::from_params<size_t>(parameters(), "dims", 10), 1, 4096);

                // resize buffers
                m_idata.resize(tensor.dims(), tensor.rows(), tensor.cols());
                m_odata.resize(odims, 1, 1);

                m_wdata.resize(1, odims, idims);
                m_bdata.resize(odims, 1, 1);

                m_gwdata.resize(1, odims, idims);
                m_gbdata.resize(odims, 1, 1);

                // create opencl objects (if available)
                ocl::manager_t& theocl = ocl::manager_t::instance();
                if (theocl.valid() && tensor.size() > 0)
                {
                        // kernels
                        m_ocl_queue = theocl.make_command_queue();
                        m_ocl_program = theocl.make_program_from_text(ocl_linear_source);
                        m_ocl_fkernel = theocl.make_kernel(m_ocl_program, "linear_forward");

                        // forward buffers
                        m_ocl_idata = theocl.make_buffer(m_idata.size() * sizeof(scalar_t), CL_MEM_READ_ONLY);
                        m_ocl_bdata = theocl.make_buffer(m_bdata.size() * sizeof(scalar_t), CL_MEM_READ_ONLY);
                        m_ocl_wdata = theocl.make_buffer(m_wdata.size() * sizeof(scalar_t), CL_MEM_READ_ONLY);
                        m_ocl_odata = theocl.make_buffer(m_odata.size() * sizeof(scalar_t), CL_MEM_WRITE_ONLY);

                        const int isize_ = static_cast<int>(isize());

                        // setup forward kernel
                        m_ocl_fkernel.setArg(0, m_ocl_idata);
                        m_ocl_fkernel.setArg(1, sizeof(int), (void*)&isize_);
                        m_ocl_fkernel.setArg(2, m_ocl_wdata);
                        m_ocl_fkernel.setArg(3, m_ocl_bdata);
                        m_ocl_fkernel.setArg(4, m_ocl_odata);
                }

                return m_wdata.size() + m_bdata.size();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void linear_layer_t::zero_params()
        {
                m_wdata.zero();
                m_bdata.zero();

                params_changed();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void linear_layer_t::random_params(scalar_t min, scalar_t max)
        {
                m_wdata.random(random_t<scalar_t>(min, max));
                m_bdata.random(random_t<scalar_t>(min, max));

                params_changed();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        ovectorizer_t& linear_layer_t::save_params(ovectorizer_t& s) const
        {
                return s << m_wdata << m_bdata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        ovectorizer_t& linear_layer_t::save_grad(ovectorizer_t& s) const
        {
                return s << m_gwdata << m_gbdata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        ivectorizer_t& linear_layer_t::load_params(ivectorizer_t& s)
        {
                s >> m_wdata >> m_bdata;

                params_changed();

                return s;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void linear_layer_t::params_changed() const
        {
                // send parameters to OpenCL device (if available)
                ocl::manager_t& theocl = ocl::manager_t::instance();
                if (theocl.valid())
                {
                        m_ocl_queue.enqueueWriteBuffer(m_ocl_bdata, CL_TRUE, 0, m_bdata.size() * sizeof(scalar_t), m_bdata.data());
                        m_ocl_queue.enqueueWriteBuffer(m_ocl_wdata, CL_TRUE, 0, m_wdata.size() * sizeof(scalar_t), m_wdata.data());
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor_t& linear_layer_t::forward(const tensor_t& input)
        {
                assert(input.dims() == m_idata.dims());
                assert(input.rows() == m_idata.rows());
                assert(input.cols() == m_idata.cols());

                m_idata.copy_from(input);

                // OpenCL version
                ocl::manager_t& theocl = ocl::manager_t::instance();
                if (theocl.valid())
                {
                        m_ocl_queue.enqueueWriteBuffer(m_ocl_idata, CL_TRUE, 0, m_idata.size() * sizeof(scalar_t), m_idata.data());

                        m_ocl_queue.enqueueNDRangeKernel(m_ocl_fkernel,
                                cl::NullRange,
                                cl::NDRange(osize()),
                                cl::NDRange(osize()));
                        m_ocl_queue.finish();

                        m_ocl_queue.enqueueReadBuffer(m_ocl_odata, CL_TRUE, 0, m_odata.size() * sizeof(scalar_t), m_odata.data());
                }

                // CPU version
                else
                {
                        _forward(m_idata.data(), isize(),
                                 m_wdata.data(), m_bdata.data(),
                                 m_odata.data(), osize());
                }

                return m_odata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor_t& linear_layer_t::backward(const tensor_t& gradient)
        {
                assert(gradient.dims() == m_odata.dims());
                assert(gradient.rows() == m_odata.rows());
                assert(gradient.cols() == m_odata.cols());

                // parameters gradient
                m_gwdata.copy_from(matrix_t(gradient.vector() * m_idata.vector().transpose()));
                m_gbdata.copy_from(gradient);

                // input gradient
                m_idata.copy_from(vector_t(m_wdata.plane_matrix(0).transpose() * gradient.vector()));

                return m_idata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}

