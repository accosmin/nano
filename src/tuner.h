#pragma once

#include "arch.h"
#include "scalar.h"
#include "stringi.h"
#include "text/json.h"

namespace nano
{
        ///
        /// \brief create the following list of scalars: offset + {1, 3} * 10^power,
        ///     where power in [min_power, max_power].
        ///
        NANO_PUBLIC scalars_t make_pow10_scalars(const scalar_t offset, const int min_power, const int max_power);

        ///
        /// \brief hyper-parameter tuning utility.
        ///
        class NANO_PUBLIC tuner_t
        {
        public:

                ///
                /// \brief hyper-parameter description.
                ///
                struct param_t
                {
                        param_t() = default;
                        param_t(const char* name, scalars_t&& values) :
                                m_name(name), m_values(values)
                        {
                        }

                        auto size() const { return m_values.size(); }
                        int precision() const { return m_precision; }
                        void precision(const int p) { m_precision = p; }

                        // attributes
                        const char*     m_name{nullptr};
                        scalars_t       m_values;
                        int             m_precision{6};
                };

                ///
                /// \brief add a new hyper-parameter to tune
                ///
                param_t& add(const char* name, scalars_t values)
                {
                        m_params.emplace_back(name, std::move(values));
                        return *m_params.rbegin();
                }

                ///
                /// \brief returns up to *max_configs* JSON configurations to evaluate
                ///
                jsons_t get(const size_t max_configs) const;

                ///
                /// \brief returns the number of parameters to tune
                ///
                auto n_params() const { return m_params.size(); }

                ///
                /// \brief returns the number of hyper-parameter configurations
                ///
                size_t n_configs() const;

        private:

                json_t json(const scalars_t&) const;
                void map(const indices_t&, scalars_t&) const;

                // attributes
                std::vector<param_t>    m_params;       ///< hyper-parameter description
        };
}
