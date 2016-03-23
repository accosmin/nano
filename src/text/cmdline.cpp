#include "cmdline.h"
#include "align.hpp"
#include "algorithm.h"
#include <cassert>
#include <fstream>
#include <iostream>

namespace nano
{
        struct option_t
        {
                explicit option_t(const std::string& short_name = std::string(),
                                  const std::string& name = std::string(),
                                  const std::string& description = std::string(),
                                  const std::string& default_value = std::string()) :
                        m_short_name(short_name),
                        m_name(name),
                        m_description(description),
                        m_default_value(default_value),
                        m_given(false)
                {
                }

                std::string concatenate() const
                {
                        return  (m_short_name.empty() ? "" : ("-" + m_short_name) + ",") +
                                "--" + m_name +
                                (m_default_value.empty() ? "" : ("(" + m_default_value + ")"));
                }

                bool has() const
                {
                        return m_given;
                }

                std::string get() const
                {
                        return m_value.empty() ? m_default_value : m_value;
                }

                // attributes
                std::string     m_short_name;
                std::string     m_name;
                std::string     m_description;
                std::string     m_default_value;
                std::string     m_value;
                bool            m_given;
        };

        bool operator==(const option_t& option, const std::string& name_or_short_name)
        {
                return  option.m_short_name == name_or_short_name ||
                        option.m_name == name_or_short_name;
        }

        using options_t = std::vector<option_t>;

        struct cmdline_t::impl_t
        {
                explicit impl_t(const std::string& title) :
                        m_title(title)
                {
                }

                auto find(const std::string& name_or_short_name)
                {
                        return std::find(m_options.begin(), m_options.end(), name_or_short_name);
                }

                auto store(const std::string& name_or_short_name, const std::string& value = std::string())
                {
                        auto it = find(name_or_short_name);
                        if (it == m_options.end())
                        {
                                log_critical("cmdline: unrecognized option [" + name_or_short_name + "]");
                        }
                        else
                        {
                                it->m_given = true;
                                it->m_value = value;
                        }
                }

                void log_critical(const std::string& message) const
                {
                        std::cout << message << std::endl << std::endl;
                        usage();
                }

                void usage() const
                {
                        std::cout << m_title << std::endl;

                        size_t max_option_size = 0;
                        for (const auto& option : m_options)
                        {
                                max_option_size = std::max(max_option_size, option.concatenate().size());
                        }

                        max_option_size += 4;
                        for (const auto& option : m_options)
                        {
                                std::cout << "  " << nano::align(option.concatenate(), max_option_size)
                                          << option.m_description << std::endl;
                       }

                        std::cout << std::endl;

                        exit(EXIT_FAILURE);
                }

                std::string     m_title;
                options_t       m_options;
        };

        cmdline_t::cmdline_t(const std::string& title) :
                m_impl(new impl_t(title))
        {
                add("h", "help", "usage");
        }

        cmdline_t::~cmdline_t() = default;

        void cmdline_t::add(const std::string& short_name, const std::string& name, const std::string& description) const
        {
                const std::string default_value;
                add(short_name, name, description, default_value);
        }

        void cmdline_t::add(
                const std::string& short_name, const std::string& name, const std::string& description,
                const std::string& default_value) const
        {
                if (    name.empty() ||
                        nano::starts_with(name, "-") ||
                        nano::starts_with(name, "--"))
                {
                        log_critical("cmdline: invalid option name [" + name + "]");
                }

                if (    !short_name.empty() &&
                        (short_name.size() != 1 || short_name[0] == '-'))
                {
                        log_critical("cmdline: invalid short option name [" + short_name + "]");
                }

                if (    m_impl->find(name) != m_impl->m_options.end())
                {
                        log_critical("cmdline: duplicated option [" + name + "]");
                }

                if (    !short_name.empty() &&
                        m_impl->find(short_name) != m_impl->m_options.end())
                {
                        log_critical("cmdline: duplicated option [" + short_name + "]");
                }

                m_impl->m_options.emplace_back(short_name, name, description, default_value);
        }

        void cmdline_t::process(const int argc, const char* argv[]) const
        {
                std::string current_name_or_short_name;

                for (int i = 1; i < argc; ++ i)
                {
                        const std::string token = argv[i];
                        assert(!token.empty());

                        if (nano::starts_with(token, "--"))
                        {
                                const std::string name = token.substr(2);

                                if (name.empty())
                                {
                                        log_critical(
                                        "cmdline: invalid option name [" + name + "/" + token + "]");
                                }

                                m_impl->store(name);
                                current_name_or_short_name = name;
                        }
                        else if (nano::starts_with(token, "-"))
                        {
                                const std::string short_name = token.substr(1);

                                if (short_name.size() != 1)
                                {
                                        log_critical(
                                        "cmdline: invalid short option name [" + short_name + "/" + token + "]");
                                }

                                m_impl->store(short_name);
                                current_name_or_short_name = short_name;
                        }
                        else
                        {
                                const std::string& value = token;

                                if (current_name_or_short_name.empty())
                                {
                                        log_critical("cmdline: missing option before value [" + value + "]");
                                }

                                m_impl->store(current_name_or_short_name, value);
                                current_name_or_short_name.clear();
                        }
                }

                if (has("help"))
                {
                        usage();
                }
        }

        void cmdline_t::process(const std::string& config) const
        {
                const auto tokens = nano::split(config, " \t\n\r");

                std::vector<const char*> ptokens(1 + tokens.size());
                ptokens[0] = nullptr;
                for (size_t i = 0; i < tokens.size(); ++ i)
                {
                        ptokens[i + 1] = tokens[i].data();
                }

                process(static_cast<int>(tokens.size() + 1), ptokens.data());
        }

        void cmdline_t::process_config_file(const std::string& path) const
        {
                std::ifstream in(path.c_str());
                const std::string config((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

                process(config);
        }

        bool cmdline_t::has(const std::string& name_or_short_name) const
        {
                const auto it = m_impl->find(name_or_short_name);
                if (it == m_impl->m_options.end())
                {
                        log_critical("cmdline: unrecognized option [" + name_or_short_name + "]");
                }
                return it->m_given;
        }

        std::string cmdline_t::get(const std::string& name_or_short_name) const
        {
                const auto it = m_impl->find(name_or_short_name);
                if (it == m_impl->m_options.end())
                {
                        log_critical("cmdline: unrecognized option [" + name_or_short_name + "]");
                }
                else if (!it->m_given && it->m_default_value.empty())
                {
                        log_critical("cmdline: no value provided for option [" + name_or_short_name + "]");
                }
                return it->get();
        }

        void cmdline_t::log_critical(const std::string& message) const
        {
                m_impl->log_critical(message);
        }

        void cmdline_t::usage() const
        {
                m_impl->usage();
        }
}
