/*
    Copyright 2017 Zheyong Fan and GPUMD development team.
    This file is part of GPUMD, licensed under GNU GPL version 3 or later.
*/

#include "keyword_dispatcher.cuh"
#include <cstring>
#include <vector>

namespace keyword_dispatcher {

namespace {

struct Entry {
  const char* keyword;
  Handler handler;
};

constexpr int kMaxEntries = 32;
Entry registry[kMaxEntries];
int registry_size = 0;

} // namespace

void register_handler(const char* keyword, Handler handler)
{
  if (keyword == nullptr || handler == nullptr) return;
  for (int i = 0; i < registry_size; ++i) {
    if (std::strcmp(registry[i].keyword, keyword) == 0) {
      registry[i].handler = handler;
      return;
    }
  }
  if (registry_size >= kMaxEntries) return;
  registry[registry_size].keyword = keyword;
  registry[registry_size].handler = handler;
  ++registry_size;
}

bool dispatch(const std::vector<std::string>& tokens, const Context& ctx)
{
  if (tokens.empty()) return false;
  for (int i = 0; i < registry_size; ++i) {
    if (tokens[0] == registry[i].keyword && registry[i].handler != nullptr) {
      registry[i].handler(tokens, ctx);
      return true;
    }
  }
  return false;
}

} // namespace keyword_dispatcher
