#pragma once

#include "arch.h"
#include "tensor.h"
#include "core/json.h"
#include "core/factory.h"

namespace nano
{
        class activation_t;
        using activation_factory_t = factory_t<activation_t>;
        using ractivation_t = activation_factory_t::trobject;

        NANO_PUBLIC activation_factory_t& get_activations();

        ///
        /// \brief non-parametric activation node:
        ///     applies a non-linear scalar function to the each scalar input,
        ///     useful for ANN models like MLPs or CNNs.
        ///
        class activation_t : public json_configurable_t
        {
        public:
                using array_t = decltype(tensor4d_map_t().array());
                using carray_t = decltype(tensor4d_cmap_t().array());

                ///
                /// \brief non-parametric activation functions
                ///
                void to_json(json_t&) const final {}
                void from_json(const json_t&) final {}

                ///
                /// \brief compute the output
                ///
                void output(tensor4d_cmap_t idata, tensor4d_map_t odata) const
                {
                        assert(idata.dims() == odata.dims());
                        output(idata.array(), odata.array());
                }

                ///
                /// \brief compute the gradient wrt the input
                ///
                void ginput(tensor4d_map_t idata, tensor4d_cmap_t odata) const
                {
                        assert(idata.dims() == odata.dims());
                        ginput(idata.array(), odata.array());
                }

        private:

                virtual void output(const carray_t& idata, array_t&& odata) const = 0;
                virtual void ginput(array_t&& idata, const carray_t& odata) const = 0;
        };

        ///
        /// \brief sin(x) activation function.
        ///
        struct activation_sine_t final : activation_t
        {
                void output(const carray_t& idata, array_t&& odata) const final
                {
                        odata = idata.sin();
                }

                void ginput(array_t&& idata, const carray_t& odata) const final
                {
                        idata = odata * idata.cos();
                }
        };

        ///
        /// \brief x/sqrt(1+x^2) activation function.
        ///
        struct activation_snorm_t final : activation_t
        {
                void output(const carray_t& idata, array_t&& odata) const final
                {
                        odata = idata * (1 + idata.square()).rsqrt();
                }

                void ginput(array_t&& idata, const carray_t& odata) const final
                {
                        idata = odata * (1 + idata.square()).cube().rsqrt();
                }
        };

        ///
        /// \brief (e^x-e^-x)/(e^x+e^-x) hyperbolic tangent activation function.
        ///
        struct activation_tanh_t final : public activation_t
        {
                void output(const carray_t& idata, array_t&& odata) const final
                {
                        odata = idata.tanh();
                }

                void ginput(array_t&& idata, const carray_t& odata) const final
                {
                        idata = odata * 4 / (idata.exp() + (-idata).exp()).square();
                }
        };

        ///
        /// \brief e^x/(1+e^x) sigmoid activation function.
        ///
        struct activation_sigm_t final : public activation_t
        {
                void output(const carray_t& idata, array_t&& odata) const final
                {
                        odata = idata.exp() / (1 + idata.exp());
                }

                void ginput(array_t&& idata, const carray_t& odata) const final
                {
                        idata = odata * idata.exp() / (1 + idata.exp()).square();
                }
        };

        ///
        /// \brief x/(1+abs(x)) soft-sign activation function.
        ///
        struct activation_ssign_t final : public activation_t
        {
                void output(const carray_t& idata, array_t&& odata) const final
                {
                        odata = idata / (1 + idata.abs());
                }

                void ginput(array_t&& idata, const carray_t& odata) const final
                {
                        idata = odata / (1 + idata.abs()).square();
                }
        };

        ///
        /// \brief log(1 + exp(x)) soft-plus activation function.
        ///
        struct activation_splus_t final : public activation_t
        {
                void output(const carray_t& idata, array_t&& odata) const final
                {
                        odata = (1 + idata.exp()).log();
                }

                void ginput(array_t&& idata, const carray_t& odata) const final
                {
                        idata = odata * idata.exp() / (1 + idata.exp());
                }
        };

        ///
        /// \brief identity activation function (for testing purposes).
        ///
        struct activation_unit_t final : public activation_t
        {
                void output(const carray_t& idata, array_t&& odata) const final
                {
                        odata = idata;
                }

                void ginput(array_t&& idata, const carray_t& odata) const final
                {
                        idata = odata;
                }
        };

        ///
        /// \brief x/(1+x^2) polynomial wave activation function.
        ///
        struct activation_pwave_t final : public activation_t
        {
                void output(const carray_t& idata, array_t&& odata) const final
                {
                        odata = idata / (1 + idata.square());
                }

                void ginput(array_t&& idata, const carray_t& odata) const final
                {
                        idata = odata * (1 - idata.square()) / (1 + idata.square()).square();
                }
        };
}
