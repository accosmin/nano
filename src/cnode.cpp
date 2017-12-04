#include "cnode.h"

using namespace nano;

cnode_t::cnode_t(const cnode_t& other) :
        m_name(other.m_name),
        m_type(other.m_type),
        m_node(other.m_node->clone()),
        m_inodes(other.m_inodes),
        m_onodes(other.m_onodes)
{
}

cnode_t::cnode_t(string_t name, string_t type, rlayer_t&& node) :
        m_name(std::move(name)),
        m_type(std::move(type)),
        m_node(std::move(node))
{
}
