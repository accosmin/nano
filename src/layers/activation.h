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

                virtual void configure(const tensor3d_dims_t&, const string_t&) override;
                virtual void output(tensor3d_const_map_t, tensor1d_const_map_t, tensor3d_map_t) override;
                virtual void ginput(tensor3d_map_t, tensor1d_const_map_t, tensor3d_const_map_t) override;
                virtual void gparam(tensor3d_const_map_t, tensor1d_map_t, tensor3d_const_map_t) override;

                virtual tensor3d_dims_t idims() const override { return m_idims; }
                virtual tensor3d_dims_t odims() const override { return m_odims; }
                virtual tensor_size_t fanin() const override { return 1; }
                virtual tensor_size_t psize() const override { return 0; }
                virtual const probe_t& probe_output() const override { return m_probe_output; }
                virtual const probe_t& probe_ginput() const override { return m_probe_ginput; }
                virtual const probe_t& probe_gparam() const override { return m_probe_gparam; }

                using tensor3d_array_t = decltype(tensor3d_map_t().array());
                using tensor3d_const_array_t = decltype(tensor3d_const_map_t().array());

        private:

                virtual void aoutput(tensor3d_const_array_t idims, tensor3d_array_t odims) const = 0;
                virtual void aginput(tensor3d_array_t idims, tensor3d_const_array_t odims) const = 0;

                // attributes
                tensor3d_dims_t m_idims;
                tensor3d_dims_t m_odims;
                probe_t         m_probe_output;
                probe_t         m_probe_ginput;
                probe_t         m_probe_gparam;
        };

        ///
        /// \brief sin(x) activation function.
        ///
        struct activation_layer_sine_t final : public activation_layer_t
        {
                using activation_layer_t::activation_layer_t;

                virtual rlayer_t clone() const override;
                virtual void aoutput(tensor3d_const_array_t, tensor3d_array_t) const override;
                virtual void aginput(tensor3d_array_t, tensor3d_const_array_t) const override;
        };

        ///
        /// \brief x/sqrt(1+x^2) activation function.
        ///
        struct activation_layer_snorm_t final : public activation_layer_t
        {
                using activation_layer_t::activation_layer_t;

                virtual rlayer_t clone() const override;
                virtual void aoutput(tensor3d_const_array_t, tensor3d_array_t) const override;
                virtual void aginput(tensor3d_array_t, tensor3d_const_array_t) const override;
        };

        ///
        /// \brief (e^x-e^-x)/(e^x+e^-x) hyperbolic tangent activation function.
        ///
        struct activation_layer_tanh_t final : public activation_layer_t
        {
                using activation_layer_t::activation_layer_t;

                virtual rlayer_t clone() const override;
                virtual void aoutput(tensor3d_const_array_t, tensor3d_array_t) const override;
                virtual void aginput(tensor3d_array_t, tensor3d_const_array_t) const override;
        };

        ///
        /// \brief e^x/(1+e^x) sigmoid activation function.
        ///
        struct activation_layer_sigm_t final : public activation_layer_t
        {
                using activation_layer_t::activation_layer_t;

                virtual rlayer_t clone() const override;
                virtual void aoutput(tensor3d_const_array_t, tensor3d_array_t) const override;
                virtual void aginput(tensor3d_array_t, tensor3d_const_array_t) const override;
        };

        ///
        /// \brief log(1 + exp(x)) soft-plus activation function.
        ///
        struct activation_layer_splus_t final : public activation_layer_t
        {
                using activation_layer_t::activation_layer_t;

                virtual rlayer_t clone() const override;
                virtual void aoutput(tensor3d_const_array_t, tensor3d_array_t) const override;
                virtual void aginput(tensor3d_array_t, tensor3d_const_array_t) const override;
        };

        ///
        /// \brief identity activation function.
        ///
        struct activation_layer_unit_t final : public activation_layer_t
        {
                using activation_layer_t::activation_layer_t;

                virtual rlayer_t clone() const override;
                virtual void aoutput(tensor3d_const_array_t, tensor3d_array_t) const override;
                virtual void aginput(tensor3d_array_t, tensor3d_const_array_t) const override;
        };

        ///
        /// \brief x/(1+x^2) polynomial wave activation function.
        ///
        struct activation_layer_pwave_t final : public activation_layer_t
        {
                using activation_layer_t::activation_layer_t;

                virtual rlayer_t clone() const override;
                virtual void aoutput(tensor3d_const_array_t, tensor3d_array_t) const override;
                virtual void aginput(tensor3d_array_t, tensor3d_const_array_t) const override;
        };
}
