#pragma once

namespace ncv
{
        namespace min
        {
                ///
                /// \brief optimization status code
                ///
                enum class status
                {
                        converged,                      ///< converged within the maximum number of iterations
                        max_iterations,                 ///< maximum number of iterations passed without convergence
                        linesearch_failed               ///< linesearch failed (e.g. numerical precision, logical error)
                };
        }
}

