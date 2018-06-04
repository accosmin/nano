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
                void to_json(json_t&) const final {};
                void from_json(const json_t&) final {};

                bool resize(const tensor3d_dims_t& idims) final;

                void random(vector_map_t pdata) const final;
                void output(tensor4d_cmaps_t idata, vector_cmap_t pdata, tensor4d_map_t odata) final;
                void ginput(tensor4d_maps_t idata, vector_cmap_t pdata, tensor4d_cmap_t odata) final;
                void gparam(tensor4d_cmaps_t idata, vector_map_t pdata, tensor4d_cmap_t odata) final;

                tensor_size_t psize() const final { return 0; }
                tensor3d_dim_t odims() const final { return m_xdims; }
                tensor_size_t flops_output() const final { return 10 * nano::size(odims()); }
                tensor_size_t flops_ginput() const final { return 10 * nano::size(odims()); }
                tensor_size_t flops_gparam() const final { return 0; }

        private:

                // attributes
                tensor3d_dim_t m_xdims{{0, 0, 0}};     ///< input/output dimensions
        };

        template <typename top>
        rlayer_t activation_layer_t<top>::clone() const
        {
                return std::make_unique<activation_layer_t<top>>(*this);
        }

        template <typename top>
        bool activation_layer_t<top>::resize(const tensor3d_dims_t& idims)
        {
                if (idims.size() != 1)
                {
                        return false;
                }

                m_xdims = idims[0];
                return true;
        }

        template <typename top>
        void activation_layer_t<top>::random(vector_map_t pdata) const
        {
                assert(pdata.size() == psize());
                NANO_UNUSED1_RELEASE(pdata);
        }

        template <typename top>
        void activation_layer_t<top>::output(tensor4d_cmaps_t idata, vector_cmap_t pdata, tensor4d_map_t odata)
        {
                assert(idata.size() == 1);
                assert(idata[0].dims() == odata.dims());
                assert(pdata.size() == psize());

                top::output(idata[0].array(), odata.array());

                NANO_UNUSED1_RELEASE(pdata);
        }

        template <typename top>
        void activation_layer_t<top>::ginput(tensor4d_maps_t idata, vector_cmap_t pdata, tensor4d_cmap_t odata)
        {
                assert(idata.size() == 1);
                assert(idata[0].dims() == odata.dims());
                assert(pdata.size() == psize());

                top::ginput(idata[0].array(), odata.array());

                NANO_UNUSED1_RELEASE(pdata);
        }

        template <typename top>
        void activation_layer_t<top>::gparam(tensor4d_cmaps_t idata, vector_map_t pdata, tensor4d_cmap_t odata)
        {
                assert(idata.size() == 1);
                assert(idata[0].dims() == odata.dims());
                assert(pdata.size() == psize());
                NANO_UNUSED3_RELEASE(idata, pdata, odata);
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
                        odata = idata * (1 + idata.square()).rsqrt();
                }

                template <typename tiarray, typename toarray>
                static void ginput(tiarray&& idata, const toarray& odata)
                {
                        idata = odata * (1 + idata.square()).cube().rsqrt();
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
        /// \brief x/(1+abs(x)) soft-sign activation function.
        ///
        struct activation_ssign_t
        {
                template <typename tiarray, typename toarray>
                static void output(const tiarray& idata, toarray&& odata)
                {
                        odata = idata / (1 + idata.abs());
                }

                template <typename tiarray, typename toarray>
                static void ginput(tiarray&& idata, const toarray& odata)
                {
                        idata = odata / (1 + idata.abs()).square();
                }
        };

        using activation_layer_ssign_t = activation_layer_t<activation_ssign_t>;

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
