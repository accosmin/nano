#pragma once

#include <algorithm>

namespace nano
{
        ///
        /// \brief set the elements of the given tensors to random values.
        ///
        template <typename trandom, typename ttensor>
        void set_random(trandom&& rgen, ttensor&& tensor)
        {
                std::for_each(tensor.data(), tensor.data() + tensor.size(), [&] (auto& v) { v = rgen(); });
        }

        template <typename trandom, typename ttensor, typename... tothers>
        void set_random(trandom&& rgen, ttensor&& tensor, tothers&&... others)
        {
                set_random(rgen, tensor);
                set_random(rgen, others...);
        }

        ///
        /// \brief add random values to the elements of the given tensors.
        ///
        template <typename trandom, typename ttensor>
        void add_random(trandom&& rgen, ttensor&& tensor)
        {
                std::for_each(tensor.data(), tensor.data() + tensor.size(), [&] (auto& v) { v += rgen(); });
        }

        template <typename trandom, typename ttensor, typename... tothers>
        void add_random(trandom&& rgen, ttensor&& tensor, tothers&&... others)
        {
                add_random(rgen, tensor);
                add_random(rgen, others...);
        }

        ///
        /// \brief check if all coefficients of the given tensor are finite.
        ///
        template <typename ttensor>
        bool isfinite(const ttensor& tensor)
        {
                const auto op = [] (const typename ttensor::Scalar v) { return !std::isfinite(v); };
                return  std::find_if(tensor.data(), tensor.data() + tensor.size(), op) ==
                        tensor.data() + tensor.size();
        }
}
