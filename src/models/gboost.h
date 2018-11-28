#pragma once

#include "core/cast.h"

namespace nano
{
        ///
        /// \brief regularization methods for Gradient Boosting.
        ///
        /// see "The Elements of Statistical Learning", by Trevor Hastie, Robert Tibshirani, Jerome Friedman
        /// see "Empirical Bernstein Boosting", by Pannagadatta K. Shivaswamy & Tony Jebara
        /// see "Variance Penalizing AdaBoost", by Pannagadatta K. Shivaswamy & Tony Jebara
        ///

        // todo: this is not needed anymore
        enum class gboost_tune
        {
                none,                   ///<
                variance,               ///< empirical variance like EBBoost/VadaBoost (needs tuning)
                shrinkage,              ///< constant shrinkage factor (needs tuning)
                stochastic,             ///< feature selection performed on a random subset (needs tuning)
        };

        template <>
        inline enum_map_t<gboost_tune> enum_string<gboost_tune>()
        {
                return
                {
                        { gboost_tune::none,            "none" },
                        { gboost_tune::variance,        "variance" },
                        { gboost_tune::shrinkage,       "shrinkage" },
                        { gboost_tune::stochastic,      "stochastic" }
                };
        }

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
        /// \brief evaluation method of the weak learner.
        ///
        enum class wlearner_eval
        {
                train,                  ///< fit and evaluate on training
                valid,                  ///< fit on training, evaluate goodness on validation
        };

        template <>
        inline enum_map_t<wlearner_eval> enum_string<wlearner_eval>()
        {
                return
                {
                        { wlearner_eval::train,         "train" },
                        { wlearner_eval::valid,         "valid" }
                };
        }

        ///
        /// \brief toggle regularization using shrinkage.
        /// NB: requires tuning the shrinkage factor.
        ///
        enum class shrinkage
        {
                on,
                off,
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
                on,
                off,
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
}
