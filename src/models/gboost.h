#pragma once

#include "core/cast.h"

namespace nano
{
        ///
        /// \brief regularization methods for Gradient Boosting.
        ///
        enum class gboost_tune
        {
                none,                   ///<
                vadaboost,              ///< VadaBoost (needs tuning)
                shrinkage_constant,     ///< constant shrinkage factor (needs tuning)
                shrinkage_geometric,    ///< geometrically decreasing shrinkage factor (needs tuning)
                stochastic,             ///< feature selection performed on a random subset (needs tuning)
        };

        template <>
        inline enum_map_t<gboost_tune> enum_string<gboost_tune>()
        {
                return
                {
                        { gboost_tune::none,                    "none" },
                        { gboost_tune::vadaboost,               "vadaboost" },
                        { gboost_tune::shrinkage_constant,      "shrink_const" },
                        { gboost_tune::shrinkage_geometric,     "shrink_geom" },
                        { gboost_tune::stochastic,              "stochastic" }
                };
        }

        ///
        /// \brief stump type.
        ///
        enum class stump_type
        {
                real,                           ///< stump \in R (no restriction)
                discrete,                       ///< stump \in {-1, +1}
        };

        template <>
        inline enum_map_t<stump_type> enum_string<stump_type>()
        {
                return
                {
                        { stump_type::real,             "real" },
                        { stump_type::discrete,         "discrete" }
                };
        }
}
