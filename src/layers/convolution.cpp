#include "convolution.h"
#include "math/numeric.h"

namespace nano
{
        convolution_layer_t::convolution_layer_t(const string_t& parameters) :
                layer_t(to_params(parameters, "dims", "16[1,4096]", "rows", "8[1,32]", "cols", "8[1,32]",
                "conn", "1[1,16]", "drow", "1[1,8]", "dcol", "1[1,8]"))
        {
        }

        rlayer_t convolution_layer_t::clone() const
        {
                return std::make_unique<convolution_layer_t>(*this);
        }

        void convolution_layer_t::configure(const tensor3d_dims_t& idims, const string_t& name)
        {
                const auto imaps = std::get<0>(idims);
                const auto irows = std::get<1>(idims);
                const auto icols = std::get<2>(idims);

                const auto omaps = clamp(from_params<tensor_size_t>(config(), "dims"), 1, 4096);
                const auto krows = clamp(from_params<tensor_size_t>(config(), "rows"), 1, 32);
                const auto kcols = clamp(from_params<tensor_size_t>(config(), "cols"), 1, 32);
                const auto kconn = clamp(from_params<tensor_size_t>(config(), "conn"), 1, 16);
                const auto drows = clamp(from_params<tensor_size_t>(config(), "drow"), 1, 8);
                const auto dcols = clamp(from_params<tensor_size_t>(config(), "dcol"), 1, 8);

                m_params = conv3d_params_t{imaps, irows, icols, omaps, kconn, krows, kcols, drows, dcols};
                if (!m_params.valid())
                {
                        throw std::invalid_argument("invalid configuration for the convolution layer");
                }

                const auto sparse = m_params.kconn() > 2 && (m_params.krows() * m_params.kcols() > 9);
                if (sparse)
                {
                        m_sparse_op = conv3d_dmaps_t{m_params};
                }
                else
                {
                        m_dense_op = conv3d_dense_t{m_params};
                }

                m_probe_output = probe_t{name, name + "(output)", m_params.flops_output()};
                m_probe_ginput = probe_t{name, name + "(ginput)", m_params.flops_ginput()};
                m_probe_gparam = probe_t{name, name + "(gparam)", m_params.flops_gparam()};
        }

        tensor_size_t convolution_layer_t::fanin() const
        {
                return m_params.krows() * m_params.kcols() * m_params.imaps() / m_params.kconn();
        }

        void convolution_layer_t::output(tensor3d_const_map_t idata, tensor1d_const_map_t param, tensor3d_map_t odata)
        {
                assert(idata.dims() == idims());
                assert(param.size() == psize());
                assert(odata.dims() == odims());

                m_probe_output.measure([&] ()
                {
                        if (m_sparse_op.params() == m_params)
                        {
                                m_sparse_op.output(idata, kdata(param), bdata(param), odata);
                        }
                        else
                        {
                                m_dense_op.output(idata, kdata(param), bdata(param), odata);
                        }
                });
        }

        void convolution_layer_t::ginput(tensor3d_map_t idata, tensor1d_const_map_t param, tensor3d_const_map_t odata)
        {
                assert(idata.dims() == idims());
                assert(param.size() == psize());
                assert(odata.dims() == odims());

                m_probe_ginput.measure([&] ()
                {
                        if (m_sparse_op.params() == m_params)
                        {
                                m_sparse_op.ginput(idata, kdata(param), bdata(param), odata);
                        }
                        else
                        {
                                m_dense_op.ginput(idata, kdata(param), bdata(param), odata);
                        }
                });
        }

        void convolution_layer_t::gparam(tensor3d_const_map_t idata, tensor1d_map_t param, tensor3d_const_map_t odata)
        {
                assert(idata.dims() == idims());
                assert(param.size() == psize());
                assert(odata.dims() == odims());

                m_probe_gparam.measure([&] ()
                {
                        if (m_sparse_op.params() == m_params)
                        {
                                m_sparse_op.gparam(idata, kdata(param), bdata(param), odata);
                        }
                        else
                        {
                                m_dense_op.gparam(idata, kdata(param), bdata(param), odata);
                        }
                });
        }
}
