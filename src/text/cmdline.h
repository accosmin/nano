#pragma once

#include "arch.h"
#include "to_string.hpp"
#include "from_string.hpp"
#include <memory>

namespace text
{
        ///
        /// \brief command line processing of the form:
        ///     --option [value]
        ///     -o [value]s
        ///
        /// other properties:
        ///     - -h,--help is built-in
        ///     - any error is considered critical and reported as an exception
        ///             (e.g. duplicated option names, missing option value, invalid option name)
        ///     - each option must have a long name, while the short name (single character) is optional
        ///     - options need not have an associated value (they can be interpreted as boolean flags)
        ///
        class ZOB_PUBLIC cmdline_t
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit cmdline_t(const std::string& title);

                ///
                /// \brief disable copying
                ///
                cmdline_t(const cmdline_t&) = delete;
                cmdline_t& operator=(const cmdline_t&) = delete;

                ///
                /// \brief destructor
                ///
                ~cmdline_t();

                ///
                /// \brief add new option by name and short name (without dash)
                ///
                void add(const std::string& short_name, const std::string& name, const std::string& description) const;

                ///
                /// \brief add new option with default value by name and short name (without dash)
                ///
                template
                <
                        typename tvalue
                >
                void add(const std::string& short_name, const std::string& name, const std::string& description,
                         const tvalue default_value) const
                {
                        add(short_name, name, description, to_string(default_value));
                }

                ///
                /// \brief process the command line arguments
                ///
                void process(const int argc, char* argv[]) const;

                ///
                /// \brief check if an option was set
                ///
                bool has(const std::string& name_or_short_name) const;

                ///
                /// \brief get the value of an option
                ///
                std::string get(const std::string& name_or_short_name) const;

                ///
                /// \brief get the value of an option as a given type
                ///
                template
                <
                        typename tvalue
                >
                tvalue get(const std::string& name_or_short_name) const
                {
                        return text::from_string<tvalue>(get(name_or_short_name));
                }

                ///
                /// \brief print help menu
                ///
                void usage() const;

        private:

                ///
                /// \brief add a new option
                ///
                void add(const std::string& short_name, const std::string& name, const std::string& description,
                         const std::string& default_value) const;

        private:

                // attributes
                struct impl_t;
                std::unique_ptr<impl_t> m_impl;
        };
}

