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
        /// \brief set the given tensors to zero.
        ///
        template <typename ttensor>
        void set_zero(ttensor&& tensor)
        {
                tensor.zero();
        }

        template <typename ttensor, typename... tothers>
        void set_zero(ttensor&& tensor, tothers&&... others)
        {
                set_zero(tensor);
                set_zero(others...);
        }

        ///
        /// \brief normalize the given tensors (t := t / t.size())
        ///
        template <typename ttensor>
        void normalize(ttensor&& tensor)
        {
                tensor /= static_cast<typename std::remove_reference<ttensor>::type::Scalar>(tensor.size());
        }

        template <typename ttensor, typename... tothers>
        void normalize(ttensor&& tensor, tothers&&... others)
        {
                normalize(tensor);
                normalize(others...);
        }
}

