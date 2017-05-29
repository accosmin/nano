#pragma once

#include "layer.h"
#include "timing.h"
#include "chrono/timer.h"

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
                timing_t        m_output_timings;
                timing_t        m_ginput_timings;
                timing_t        m_gparam_timings;
        };

        using layer_infos_t = std::vector<layer_info_t>;

        inline layer_info_t::layer_info_t(const string_t& name, rlayer_t layer) :
                m_name(name), m_layer(std::move(layer))
        {
        }

        inline layer_info_t::layer_info_t(const layer_info_t& other) :
                m_name(other.m_name),
                m_layer(other.m_layer->clone()),
                m_output_timings(other.m_output_timings),
                m_ginput_timings(other.m_ginput_timings),
                m_gparam_timings(other.m_gparam_timings)
        {
        }

        inline void layer_info_t::output(const scalar_t* idata, const scalar_t* param, scalar_t* odata)
        {
                const timer_t timer;
                m_layer->output(map_tensor(idata, idims()), map_tensor(param, psize()), map_tensor(odata, odims()));
                m_output_timings(static_cast<size_t>(timer.microseconds().count()));
        }

        inline void layer_info_t::ginput(scalar_t* idata, const scalar_t* param, const scalar_t* odata)
        {
                const timer_t timer;
                m_layer->ginput(map_tensor(idata, idims()), map_tensor(param, psize()), map_tensor(odata, odims()));
                m_ginput_timings(static_cast<size_t>(timer.microseconds().count()));
        }

        inline void layer_info_t::gparam(const scalar_t* idata, scalar_t* param, const scalar_t* odata)
        {
                const timer_t timer;
                m_layer->gparam(map_tensor(idata, idims()), map_tensor(param, psize()), map_tensor(odata, odims()));
                m_gparam_timings(static_cast<size_t>(timer.microseconds().count()));
        }
}
