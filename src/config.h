#pragma once

#include "digraph.h"
#include "text/cast.h"
#include "text/json_reader.h"
#include "text/json_writer.h"

namespace nano
{
        class config_t; // as a graph
        // todo: get value for key
        // todo: traverse the digraph of key-values
        // todo: update configurable_t to use config_t
        // todo: remove to_params/from_params

        config_t from_json(const string_t& json);
        string_t to_json(const config_t& config);

        ///
        /// \brief configuration directed graph of key-value strings.
        ///
        class config_t
        {
        public:

                ///
                /// \brief insert a key-value pair and link it to its parent (if given)
                ///
                template <typename tkey, typename tvalue>
                void pair(const size_t id_parent, const tkey& key, const tvalue& val)
                {
                        const auto id_key = token(key);
                        const auto id_value = token(val);
                        connect(id_parent, id_key);
                        connect(id_key, id_value);
                }

                ///
                /// \brief insert a set of key-value pairs and link them to their parent (if given)
                ///
                template <typename tkey, typename tvalue, typename... tkeys_and_values>
                void pairs(const size_t id_parent, const tkey& key, const tvalue& val, const tkeys_and_values&... keys_and_vals)
                {
                        pair(id_parent, key, val);
                        pairs(id_parent, keys_and_vals...);
                }

                ///
                /// \brief insert a set of values and link them to their parent (if given)
                ///
                template <typename tvalue, typename... tvalues>
                void values(const size_t id_parent, const tvalue& val, const tvalues&... vals)
                {
                        value(id_parent, val);
                        if (sizeof...(vals) > 0)
                        {
                                values(id_parent, vals...);
                        }
                }

                ///
                /// \brief insert a value and link it to its parent (if given)
                ///
                template <typename tvalue>
                size_t value(const size_t id_parent, const tvalue& val)
                {
                        const auto id_value = token(val);
                        connect(id_parent, id_value);
                        return id_value;
                }

                ///
                /// \brief retrieve the associated value (if exists) for the given key chain
                /// an exception is thrown:
                ///     - if the key chain is not valid
                ///     - or the value itself is not of the right type
                ///
                template <typename tvalue, typename... tkeys>
                tvalue get(const tkeys&... keys) const
                {
                        const auto index = find(keys...);
                        return from_string<tvalue>(m_tokens.at(index));
                }

                ///
                /// \brief special indices
                ///
                size_t none() const { return string_t::npos; }
                size_t last() const { return m_tokens.size() - 1; }
                size_t next() const { return m_tokens.size(); }

        private:

                void pairs(const size_t) const {}
                void values(const size_t) const {}

                template <typename tvalue>
                size_t token(const tvalue& val)
                {
                        const size_t id = m_tokens.size();
                        m_tokens.push_back(to_string(val));
                        return id;
                }

                bool connect(const size_t src, const size_t dst)
                {
                        return src < m_tokens.size() && dst < m_tokens.size() && m_digraph.edge(src, dst);
                }

                size_t search(const size_t src) const
                {
                        const auto dsts = m_digraph.outgoing(src);
                        return dsts.size() == 1 ? dsts[0] : m_tokens.size();
                }

                template <typename tkey, typename... tkeys>
                size_t search(const size_t src, const tkey& key, const tkeys&... keys) const
                {
                        for (const auto dst : m_digraph.outgoing(src))
                        {
                                if (m_tokens[dst] == key)
                                {
                                        return search(dst, keys...);
                                }
                        }

                        return m_tokens.size();
                }

                template <typename tkey, typename... tkeys>
                size_t find(const tkey& key, const tkeys&... keys) const
                {
                        for (const auto in : m_digraph.incoming())
                        {
                                if (m_tokens[in] == key)
                                {
                                        return search(in, keys...);
                                }
                        }

                        return m_tokens.size();
                }

                // attributes
                strings_t               m_tokens;
                digraph_t<size_t>       m_digraph;
        };
}
