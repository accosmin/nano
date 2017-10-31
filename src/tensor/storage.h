#pragma once

#include "index.h"

namespace nano
{
        ///
        /// \brief tensor storage using an Eigen vector.
        /// NB: the tensor owns the allocated memory and as such the tensor is resizable.
        ///
        template <typename tscalar_>
        class tensor_vstorage_t
        {
        public:
                using tscalar = typename std::remove_const<tscalar_>::type;
                using tstorage = tensor_vector_t<tscalar>;
                using treference = tscalar&;
                using tconst_reference = const tscalar&;

                static constexpr bool resizable() { return true; }
                static constexpr bool owns_memory() { return true; }
                static constexpr bool only_const() { return false; }

                tensor_vstorage_t() = default;
                tensor_vstorage_t(const tstorage& data) : m_data(data) {}
                tensor_vstorage_t(const tensor_size_t size) : m_data(size) {}

                auto size() const { return m_data.size(); }
                void resize(const tensor_size_t size) { m_data.resize(size); }

                auto data() { return m_data.data(); }
                auto data() const { return m_data.data(); }

        private:

                // attributes
                tstorage        m_data;         ///< store tensor as a 1D vector.
        };

        ///
        /// \brief tensor storage using a C-array.
        /// NB: the tensors doesn't own the allocated memory and as such is not resizable.
        ///
        template <typename tscalar_>
        class tensor_pstorage_t
        {
        public:
                using tscalar = typename std::remove_const<tscalar_>::type;
                using tstorage = tscalar_*;
                using treference = typename std::conditional<std::is_const<tscalar_>::value, const tscalar&, tscalar&>::type;
                using tconst_reference = treference;

                static constexpr bool resizable() { return false; }
                static constexpr bool owns_memory() { return false; }
                static constexpr bool only_const() { return true; }

                tensor_pstorage_t() : m_data(nullptr) {}
                tensor_pstorage_t(const tstorage& data) : m_data(data) {}
                tensor_pstorage_t(const tensor_size_t);

                auto size() const;
                void resize(const tensor_size_t);

                auto data() { return m_data; }
                auto data() const { return m_data; }

        private:

                // attributes
                tstorage const  m_data;         ///< wrap tensor over a contiguous array.
        };
}
