#pragma once

namespace min
{
        ///
        /// \brief optimization status code
        ///
        enum class status
        {
                converged,                      ///< converged within the maximum number of iterations
                max_iterations,                 ///< maximum number of iterations passed without convergence

                ls_failed_not_descent,          ///< linesearch failed: chosen direction not a descent direction
                ls_failed_invalid_initial_step, ///< linesearch failed: negative initial step length
                ls_failed_invalid_step,         ///< linesearch failed: found infinite step length
                ls_failed_not_decreasing_step,  ///< linesearch failed: found step length that increases the function's value
        };
}

