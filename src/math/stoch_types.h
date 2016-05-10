#pragma once

namespace nano
{
        ///
        /// \brief stochastic optimization methods
        ///
        enum class stoch_optimizer
        {
                SG,                             ///< stochastic gradient
                SNG,                            ///< stochastic normalized gradient
                SGM,                            ///< stochastic gradient with momentum
                AG,                             ///< Nesterov's accelerated gradient
                AGFR,                           ///< Nesterov's accelerated gradient (with function value-based restarts)
                AGGR,                           ///< Nesterov's accelerated gradient (with gradient-based restarts)
                ADAGRAD,                        ///< AdaGrad
                ADADELTA,                       ///< AdaDelta
                ADAM                            ///< Adam
        };
}

