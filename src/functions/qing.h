#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief Qing function: see http://benchmarkfcns.xyz/benchmarkfcns/qingfcn.html.
        ///
        class function_qing_t final : public function_t
        {
        public:

                explicit function_qing_t(const tensor_size_t dims);

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override;
        };
}
