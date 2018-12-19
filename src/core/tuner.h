#pragma once

#include "arch.h"
#include "scalar.h"
#include "core/json.h"

namespace nano
{
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
                        param_t(const char* name, strings_t&& values) : m_name(name), m_values(values) {}

                        // attributes
                        const char*     m_name{nullptr};
                        strings_t       m_values;
                };

                ///
                /// \brief add a hyper-parameter to tune: select from enumeration values
                ///
                template <typename tenum>
                void add_enum(const char* name)
                {
                        strings_t values;
                        for (const auto& elem : enum_string<tenum>())
                        {
                                // cppcheck-suppress useStlAlgorithm
                                values.push_back(elem.second);
                        }
                        m_params.emplace_back(name, std::move(values));
                }

                ///
                /// \brief add a hyper-parameter to tune: select from a finite list of power of 10 scalars
                /// NB: this uses the following list of scalars: offset + {1, 3} * 10^power,
                ///     where power in [min_power, max_power].
                ///
                void add_pow10s(const char* name, const scalar_t offset, const int min_power, const int max_power);

                ///
                /// \brief add a hyper-parameter to tune: select from a finite list of scalars
                ///
                template <typename... tscalar>
                void add_finite(const char* name, tscalar... scalars)
                {
                        strings_t values;
                        for (const auto scalar : {static_cast<scalar_t>(scalars)...})
                        {
                                // cppcheck-suppress useStlAlgorithm
                                values.push_back(to_string(scalar));
                        }
                        m_params.emplace_back(name, std::move(values));
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

                json_t json(const strings_t&) const;
                void map(const indices_t&, strings_t&) const;

                // attributes
                std::vector<param_t>    m_params;       ///< hyper-parameter description
        };
}
