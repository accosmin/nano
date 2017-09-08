#pragma once

#include "tensor_index.h"

namespace nano
{
        template <typename tstorage>
        struct tensor_storage_t;

        ///
        /// \brief storage for a tensor that owns the allocated memory.
        ///
        template <typename tscalar_>
        struct tensor_storage_t<tensor_vector_t<typename std::remove_const<tscalar_>::type>
        {
                using tscalar = typename std::remove_const<tscalar_>::type;
                using tstorage = tensor_vector_t<tscalar>;
                using treference = tscalar&;
                using tconst_reference = const tscalar&;

                static constexpr bool resizable = true;
                static constexpr bool owns_memory = true;
                static constexpr bool only_const = false;

                tensor_storage_t() {}
                tensor_storage_t(const tstorage& data) : m_data(data) {}
                tensor_storage_t(const tensor_size_t size) : m_data(size) {}

                void resize(const tensor_size_t size) { m_data.resize(size); }

                auto data() { return m_data.data(); }
                auto data() const { return m_data.data(); }

        private:

                // attributes
                tstorage        m_data;         ///< store tensor as a 1D vector.
        };

        ///
        /// \brief storage for a tensor that maps the allocated memory.
        ///
        template <typename tpointer>
        struct tensor_storage_t<tpointer>
        {
                using tscalar = typename std::remove_const<typename std::remove_pointer<tpointer>::type>::type;
                using tstorage = tpointer;
                using treference = typename std::conditional<std::is_const<tpointer>::value, const tscalar&, tscalar&>::type;
                using tconst_reference = treference;

                static constexpr bool resizable = false;
                static constexpr bool owns_memory = false;
                static constexpr bool only_const = std::is_const<tpointer>::value;

                tensor_storage_t() : m_data(nullptr) {}
                tensor_storage_t(const tstorage& data) : m_data(data) {}
                tensor_storage_t(const tensor_size_t);

                void resize(const tensor_size_t);

                auto data() { return m_data; }
                auto data() const { return m_data; }

        private:

                // attributes
                tstorage        m_data;         ///< wrap tensor over a contiguous array.
        };
}
