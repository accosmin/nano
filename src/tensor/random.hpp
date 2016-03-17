#pragma once

#include <algorithm>

namespace tensor
{
        ///
        /// \brief set the elements of the given tensor to random values.
        ///
        template
        <
                typename ttensor,
                typename trandom
        >
        void set_random(ttensor&& tensor, trandom rgen)
        {
                rgen(tensor.data(), tensor.data() + tensor.size());
        }

        ///
        /// \brief add random values to the elements of the given tensor.
        ///
        template
        <
                typename ttensor,
                typename trandom
        >
        void add_random(ttensor&& tensor, trandom rgen)
        {
                std::for_each(tensor.data(), tensor.data() + tensor.size(), [&] (auto& v) { v += rgen(); });
        }
}
