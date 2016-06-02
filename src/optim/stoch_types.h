#pragma once

#include "text/enum_string.hpp"

namespace nano
{
        ///
        /// \brief stochastic optimization methods
        ///
        enum class stoch_optimizer
        {
                SG,                             ///< stochastic gradient (descent)
                NGD,                            ///< normalized gradient descent
                SGM,                            ///< stochastic gradient with momentum
                AG,                             ///< Nesterov's accelerated gradient
                AGFR,                           ///< Nesterov's accelerated gradient (with function value-based restarts)
                AGGR,                           ///< Nesterov's accelerated gradient (with gradient-based restarts)
                ADAGRAD,                        ///< AdaGrad
                ADADELTA,                       ///< AdaDelta
                ADAM                            ///< Adam
        };

        template <>
        inline std::map<stoch_optimizer, std::string> enum_string<stoch_optimizer>()
        {
                return
                {
                        { stoch_optimizer::SG,           "sg" },
                        { stoch_optimizer::NGD,          "ngd" },
                        { stoch_optimizer::SGM,          "sgm" },
                        { stoch_optimizer::AG,           "ag" },
                        { stoch_optimizer::AGFR,         "agfr" },
                        { stoch_optimizer::AGGR,         "aggr" },
                        { stoch_optimizer::ADAGRAD,      "adagrad" },
                        { stoch_optimizer::ADADELTA,     "adadelta" },
                        { stoch_optimizer::ADAM,         "adam" }
                };
        }
}

