#pragma once

#include "layer.h"
#include "chrono/probe.h"

namespace nano
{
        ///
        /// \brief wrapper a layer to provide extra information (e.g. timing per operation).
        ///
        struct layer_info_t
        {
                layer_info_t(const string_t& name = string_t(), rlayer_t layer = rlayer_t());
                layer_info_t(const layer_info_t& other);
                layer_info_t(layer_info_t&&) = default;
                layer_info_t& operator=(layer_info_t&&) = default;
                layer_info_t& operator=(const layer_info_t&) = delete;

                void output(const scalar_t* idata, const scalar_t* param, scalar_t* odata);
                void ginput(scalar_t* idata, const scalar_t* param, const scalar_t* odata);
                void gparam(const scalar_t* idata, scalar_t* param, const scalar_t* odata);

                auto idims() const { return m_layer->idims(); }
                auto odims() const { return m_layer->odims(); }
                auto isize() const { return nano::size(idims()); }
                auto osize() const { return nano::size(odims()); }
                auto xsize() const { return isize() + osize(); }
                auto psize() const { return m_layer->psize(); }
                auto flops() const { return m_layer->flops(); }

                string_t        m_name;
                rlayer_t        m_layer;
                probe_t         m_output_probe;
                probe_t         m_ginput_probe;
                probe_t         m_gparam_probe;
        };

        using layer_infos_t = std::vector<layer_info_t>;

        inline layer_info_t::layer_info_t(const string_t& name, rlayer_t layer) :
                m_name(name), m_layer(std::move(layer)),
                m_output_probl(name + "(output)"),
                m_ginput_probl(name + "(ginput)"),
                m_gparam_probl(name + "(gparam)"),
        {
        }

        inline layer_info_t::layer_info_t(const layer_info_t& other) :
                m_name(other.m_name),
                m_layer(other.m_layer->clone()),
                m_output_probe(other.m_output_probe),
                m_ginput_probe(other.m_ginput_probe),
                m_gparam_probe(other.m_gparam_probe)
        {
        }

        inline void layer_info_t::output(const scalar_t* idata, const scalar_t* param, scalar_t* odata)
        {
                m_output_probe.measure([&] ()
                {
                        m_layer->output(map_tensor(idata, idims()), map_tensor(param, psize()), map_tensor(odata, odims()));
                });
        }

        inline void layer_info_t::ginput(scalar_t* idata, const scalar_t* param, const scalar_t* odata)
        {
                m_ginput_probe.measure([&] ()
                {
                        m_layer->ginput(map_tensor(idata, idims()), map_tensor(param, psize()), map_tensor(odata, odims()));
                });
        }

        inline void layer_info_t::gparam(const scalar_t* idata, scalar_t* param, const scalar_t* odata)
        {
                m_gparam_probe.measure([&] ()
                {
                        m_layer->gparam(map_tensor(idata, idims()), map_tensor(param, psize()), map_tensor(odata, odims()));
                });
        }
}
