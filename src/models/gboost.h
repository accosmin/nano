#pragma once

#include "core/cast.h"

namespace nano
{
        ///
        /// \brief weak learner type.
        ///
        enum class wlearner_type
        {
                real,                   ///< output \in R (no restriction)
                discrete,               ///< output \in {-1, +1} (useful for classification to reduce overfitting)
        };

        template <>
        inline enum_map_t<wlearner_type> enum_string<wlearner_type>()
        {
                return
                {
                        { wlearner_type::real,          "real" },
                        { wlearner_type::discrete,      "discrete" }
                };
        }

        ///
        /// \brief toggle regularization using shrinkage.
        /// NB: requires tuning the shrinkage factor.
        ///
        enum class shrinkage
        {
                on,                     ///<
                off,                    ///<
        };

        template <>
        inline enum_map_t<shrinkage> enum_string<shrinkage>()
        {
                return
                {
                        { shrinkage::on,                "on" },
                        { shrinkage::off,               "off" }
                };
        }

        ///
        /// \brief toggle subsampling for fitting weak learners.
        /// NB: requires tuning the sampling percentage.
        ///
        /// see "Stochastic gradient boosting", by Jerome Friedman
        ///
        enum class subsampling
        {
                on,                     ///<
                off,                    ///<
        };

        template <>
        inline enum_map_t<subsampling> enum_string<subsampling>()
        {
                return
                {
                        { subsampling::on,              "on" },
                        { subsampling::off,             "off" }
                };
        }

        ///
        /// \brief cumulated loss to minimize during boosting.
        /// NB: the variance version requires tuning.
        /// see "Empirical Bernstein Boosting", by Pannagadatta K. Shivaswamy & Tony Jebara
        /// see "Variance Penalizing AdaBoost", by Pannagadatta K. Shivaswamy & Tony Jebara
        ///
        enum class cumloss
        {
                average,                ///< minimize the empirical expectation of the loss
                variance                ///< regularize the empirical expectation of the loss with its variance
        };

        template <>
        inline enum_map_t<cumloss> enum_string<cumloss>()
        {
                return
                {
                        { cumloss::average,             "avg" },
                        { cumloss::variance,            "var" }
                };
        }
}
