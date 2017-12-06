#pragma once

#include "layer.h"
#include "tensor/numeric.h"

namespace nano
{
        ///
        /// \brief activation layer: applies a non-linear scalar function to the each scalar input.
        ///
        template <typename top>
        class activation_layer_t final : public layer_t
        {
        public:

                rlayer_t clone() const final;
                json_reader_t& config(json_reader_t& reader) final { return reader; }
                json_writer_t& config(json_writer_t& writer) const final { return writer; }

                bool resize(const tensor3d_dims_t& idims, const string_t& name) final;

                void output(const tensor4d_cmap_t& idata, const vector_cmap_t& pdata, tensor4d_map_t&& odata) final;
                void ginput(tensor4d_map_t&& idata, const vector_cmap_t& pdata, const tensor4d_cmap_t& odata) final;
                void gparam(const tensor4d_cmap_t& idata, vector_map_t&& pdata, const tensor4d_cmap_t& odata) final;

                tensor_size_t fanin() const final { return 1; }
                tensor_size_t psize() const final { return 0; }
                tensor3d_dim_t idims() const final { return m_xdims; }
                tensor3d_dim_t odims() const final { return m_xdims; }

                const probe_t& probe_output() const final { return m_probe_output; }
                const probe_t& probe_ginput() const final { return m_probe_ginput; }
                const probe_t& probe_gparam() const final { return m_probe_gparam; }

        private:

                // attributes
                tensor3d_dim_t m_xdims{{0, 0, 0}};     ///< input/output dimensions
                probe_t         m_probe_output;
                probe_t         m_probe_ginput;
                probe_t         m_probe_gparam;
        };

        template <typename top>
        rlayer_t activation_layer_t<top>::clone() const
        {
                return std::make_unique<activation_layer_t<top>>(*this);
        }

        template <typename top>
        bool activation_layer_t<top>::resize(const tensor3d_dims_t& idims, const string_t& name)
        {
                if (idims.size() != 1)
                {
                        return false;
                }

                m_xdims = idims[0];
                m_probe_output = probe_t{name, name + "(output)", 10 * isize()};
                m_probe_ginput = probe_t{name, name + "(ginput)", 10 * isize()};
                m_probe_gparam = probe_t{name, name + "(gparam)", 0};
                return true;
        }

        template <typename top>
        void activation_layer_t<top>::output(const tensor4d_cmap_t& idata, const vector_cmap_t&, tensor4d_map_t&& odata)
        {
                assert(idata.dims() == odata.dims());
                assert(nano::isfinite(idata));

                const auto count = idata.size<0>();
                m_probe_output.measure([&] () { top::output(idata.array(), odata.array()); }, count);

                assert(nano::isfinite(odata));
        }

        template <typename top>
        void activation_layer_t<top>::ginput(tensor4d_map_t&& idata, const vector_cmap_t&, const tensor4d_cmap_t& odata)
        {
                assert(idata.dims() == odata.dims());
                assert(nano::isfinite(idata));
                assert(nano::isfinite(odata));

                const auto count = idata.size<0>();
                m_probe_ginput.measure([&] () { top::ginput(idata.array(), odata.array()); }, count);

                assert(nano::isfinite(idata));
        }

        template <typename top>
        void activation_layer_t<top>::gparam(const tensor4d_cmap_t&, vector_map_t&&, const tensor4d_cmap_t&)
        {
        }

        ///
        /// \brief sin(x) activation function.
        ///
        struct activation_sine_t
        {
                template <typename tiarray, typename toarray>
                static void output(const tiarray& idata, toarray&& odata)
                {
                        odata = idata.sin();
                }

                template <typename tiarray, typename toarray>
                static void ginput(tiarray&& idata, const toarray& odata)
                {
                        idata = odata * idata.cos();
                }
        };

        using activation_layer_sine_t = activation_layer_t<activation_sine_t>;

        ///
        /// \brief x/sqrt(1+x^2) activation function.
        ///
        struct activation_snorm_t
        {
                template <typename tiarray, typename toarray>
                static void output(const tiarray& idata, toarray&& odata)
                {
                        odata = idata / (1 + idata.square()).sqrt();
                }

                template <typename tiarray, typename toarray>
                static void ginput(tiarray&& idata, const toarray& odata)
                {
                        idata = odata / (1 + idata.square()).cube().sqrt();
                }
        };

        using activation_layer_snorm_t = activation_layer_t<activation_snorm_t>;

        ///
        /// \brief (e^x-e^-x)/(e^x+e^-x) hyperbolic tangent activation function.
        ///
        struct activation_tanh_t
        {
                template <typename tiarray, typename toarray>
                static void output(const tiarray& idata, toarray&& odata)
                {
                        odata = idata.tanh();
                }

                template <typename tiarray, typename toarray>
                static void ginput(tiarray&& idata, const toarray& odata)
                {
                        idata = odata * 4 / (idata.exp() + (-idata).exp()).square();
                }
        };

        using activation_layer_tanh_t = activation_layer_t<activation_tanh_t>;

        ///
        /// \brief e^x/(1+e^x) sigmoid activation function.
        ///
        struct activation_sigm_t
        {
                template <typename tiarray, typename toarray>
                static void output(const tiarray& idata, toarray&& odata)
                {
                        odata = idata.exp() / (1 + idata.exp());
                }

                template <typename tiarray, typename toarray>
                static void ginput(tiarray&& idata, const toarray& odata)
                {
                        idata = odata * idata.exp() / (1 + idata.exp()).square();
                }
        };

        using activation_layer_sigm_t = activation_layer_t<activation_sigm_t>;

        ///
        /// \brief log(1 + exp(x)) soft-plus activation function.
        ///
        struct activation_splus_t
        {
                template <typename tiarray, typename toarray>
                static void output(const tiarray& idata, toarray&& odata)
                {
                        odata = (1 + idata.exp()).log();
                }

                template <typename tiarray, typename toarray>
                static void ginput(tiarray&& idata, const toarray& odata)
                {
                        idata = odata * idata.exp() / (1 + idata.exp());
                }
        };

        using activation_layer_splus_t = activation_layer_t<activation_splus_t>;

        ///
        /// \brief identity activation function.
        ///
        struct activation_unit_t
        {
                template <typename tiarray, typename toarray>
                static void output(const tiarray& idata, toarray&& odata)
                {
                        odata = idata;
                }

                template <typename tiarray, typename toarray>
                static void ginput(tiarray&& idata, const toarray& odata)
                {
                        idata = odata;
                }
        };

        using activation_layer_unit_t = activation_layer_t<activation_unit_t>;

        ///
        /// \brief x/(1+x^2) polynomial wave activation function.
        ///
        struct activation_pwave_t
        {
                template <typename tiarray, typename toarray>
                static void output(const tiarray& idata, toarray&& odata)
                {
                        odata = idata / (1 + idata.square());
                }

                template <typename tiarray, typename toarray>
                static void ginput(tiarray&& idata, const toarray& odata)
                {
                        idata = odata * (1 - idata.square()) / (1 + idata.square()).square();
                }
        };

        using activation_layer_pwave_t = activation_layer_t<activation_pwave_t>;
}
