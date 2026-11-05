/*
    Copyright 2017 Zheyong Fan and GPUMD development team.
    This file is part of GPUMD, licensed under GNU GPL version 3 or later.
*/

/*----------------------------------------------------------------------------80
Keyword extension dispatcher.

Allows new run.in keywords to be added by external modules (such as the
hessian-only additions) without modifying the existing if/else dispatch
chain in src/main_gpumd/run.cu.  Modules register a handler at static
initialization time; Run::parse_one_keyword falls back to the dispatcher
once it has exhausted its built-in keywords.
------------------------------------------------------------------------------*/

#pragma once
#include <string>
#include <vector>

class Atom;
class Box;
class Force;
class Group;

namespace keyword_dispatcher {

struct Context {
  Force& force;
  Box& box;
  Atom& atom;
  std::vector<Group>& group;
  bool has_replicate;
  int replicate_size[3];
};

using Handler = void (*)(const std::vector<std::string>& tokens, const Context& ctx);

void register_handler(const char* keyword, Handler handler);

// Returns true if a registered handler consumed the keyword.
bool dispatch(const std::vector<std::string>& tokens, const Context& ctx);

} // namespace keyword_dispatcher
