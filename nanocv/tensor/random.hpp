#pragma once

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief set the elements of the given tensor to random values using the given random number generator
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
        }
}
