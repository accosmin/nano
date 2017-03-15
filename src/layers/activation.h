#pragma once

#include "layer.h"

namespace nano
{
        ///
        /// \brief activation layer: applies a non-linear scalar function to the each input.
        ///
        struct activation_layer_t : public layer_t
        {
                explicit activation_layer_t(const string_t& parameters = string_t());

                virtual void configure(const tensor3d_dims_t&) override;
                virtual void output(tensor3d_map_t, tensor1d_map_t, tensor3d_map_t) override;
                virtual void ginput(tensor3d_map_t, tensor1d_map_t, tensor3d_map_t) override;
                virtual void gparam(tensor3d_map_t, tensor1d_map_t, tensor3d_map_t) override;

                virtual dim3d_t idims() const override { return m_idims; }
                virtual dim3d_t odims() const override { return m_odims; }
                virtual tensor_size_t psize() const override { return 0; }
                virtual tensor_size_t flops() const override { return 10 * nano::size(m_idims); }

                using tensor3d_array_t = decltype(tensor3d_t(0, 0, 0).array());

        private:

                virtual void aoutput(tensor3d_array_t idims, tensor3d_array_t odims) const = 0;
                virtual void aginput(tensor3d_array_t idims, tensor3d_array_t odims) const = 0;

                // attributes
                dim3d_t         m_idims;
                dim3d_t         m_odims;
        };

        ///
        /// \brief sin(x) activation function.
        ///
        struct activation_layer_sine_t final : public activation_layer_t
        {
                using activation_layer_t::activation_layer_t;

                virtual rlayer_t clone() const override;
                virtual void aoutput(tensor3d_array_t, tensor3d_array_t) const override;
                virtual void aginput(tensor3d_array_t, tensor3d_array_t) const override;
        };

        ///
        /// \brief x/sqrt(1+x^2) activation function.
        ///
        struct activation_layer_snorm_t final : public activation_layer_t
        {
                using activation_layer_t::activation_layer_t;

                virtual rlayer_t clone() const override;
                virtual void aoutput(tensor3d_array_t, tensor3d_array_t) const override;
                virtual void aginput(tensor3d_array_t, tensor3d_array_t) const override;
        };

        ///
        /// \brief (e^x-e^-x)/(e^x+e^-x) hyperbolic tangent activation function.
        ///
        struct activation_layer_tanh_t final : public activation_layer_t
        {
                using activation_layer_t::activation_layer_t;

                virtual rlayer_t clone() const override;
                virtual void aoutput(tensor3d_array_t, tensor3d_array_t) const override;
                virtual void aginput(tensor3d_array_t, tensor3d_array_t) const override;
        };

        ///
        /// \brief e^x/(1+e^x) sigmoid activation function.
        ///
        struct activation_layer_sigm_t final : public activation_layer_t
        {
                using activation_layer_t::activation_layer_t;

                virtual rlayer_t clone() const override;
                virtual void aoutput(tensor3d_array_t, tensor3d_array_t) const override;
                virtual void aginput(tensor3d_array_t, tensor3d_array_t) const override;
        };

        ///
        /// \brief log(1 + exp(x)) soft-plus activation function.
        ///
        struct activation_layer_splus_t final : public activation_layer_t
        {
                using activation_layer_t::activation_layer_t;

                virtual rlayer_t clone() const override;
                virtual void aoutput(tensor3d_array_t, tensor3d_array_t) const override;
                virtual void aginput(tensor3d_array_t, tensor3d_array_t) const override;
        };

        ///
        /// \brief identity activation function.
        ///
        struct activation_layer_unit_t final : public activation_layer_t
        {
                using activation_layer_t::activation_layer_t;

                virtual rlayer_t clone() const override;
                virtual void aoutput(tensor3d_array_t, tensor3d_array_t) const override;
                virtual void aginput(tensor3d_array_t, tensor3d_array_t) const override;
        };

        ///
        /// \brief x/(exp(-alpha*x)+exp(+alpha*x)) exponential wave activation function.
        ///
        struct activation_layer_ewave_t final : public activation_layer_t
        {
                explicit activation_layer_ewave_t(const string_t& parameters = string_t());

                virtual rlayer_t clone() const override;
                virtual void aoutput(tensor3d_array_t, tensor3d_array_t) const override;
                virtual void aginput(tensor3d_array_t, tensor3d_array_t) const override;
        };
}
