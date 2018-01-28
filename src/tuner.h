#pragma once

#include "arch.h"
#include "scalar.h"
#include "stringi.h"

namespace nano
{
        ///
        /// \brief hyper-parameter tuning utilities.
        ///
        class NANO_PUBLIC tuner_t
        {
        public:

                struct param_t
                {
                        const char*     m_name{nullptr};
                        scalar_t        m_min{0};
                        scalar_t        m_max{0};
                        scalar_t        m_base{1};
                };

                struct trial_t
                {
                        scalars_t       m_values;       ///< value for each parameter
                        scalar_t        m_score{0};     ///< score (aka goodness) of the configuration
                };

                ///
                /// \brief add a new hyper-parameter to tune
                ///
                template <typename tmin, typename tmax, typename tbase>
                void add(const char* name, const tmin min, const tmax max, const tbase base)
                {
                        m_params.push_back({name,
                                static_cast<scalar_t>(min),
                                static_cast<scalar_t>(max),
                                static_cast<scalar_t>(base)});
                }

                ///
                /// \brief retrive a JSON configuration of hyper-parameters to evaluate
                ///
                string_t get();

                ///
                /// \brief assign a score to the last evaluated configuration
                ///
                void score(const scalar_t score);

        private:

                string_t json(const trial_t& trial) const;

                // attributes
                std::vector<param_t>    m_params;
                std::vector<trial_t>    m_trials;       ///< hyper-parameter trials evaluated so far
        };
}
