// Copyright 2021 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "envpool/core/py_envpool.h"
#include "envpool/toy_text/blackjack.h"
#include "envpool/toy_text/catch.h"
#include "envpool/toy_text/cliffwalking.h"
#include "envpool/toy_text/frozen_lake.h"
#include "envpool/toy_text/nchain.h"
#include "envpool/toy_text/taxi.h"

using CatchEnvSpec = PyEnvSpec<toy_text::CatchEnvSpec>;
using CatchEnvPool = PyEnvPool<toy_text::CatchEnvPool>;

using FrozenLakeEnvSpec = PyEnvSpec<toy_text::FrozenLakeEnvSpec>;
using FrozenLakeEnvPool = PyEnvPool<toy_text::FrozenLakeEnvPool>;

using TaxiEnvSpec = PyEnvSpec<toy_text::TaxiEnvSpec>;
using TaxiEnvPool = PyEnvPool<toy_text::TaxiEnvPool>;

using NChainEnvSpec = PyEnvSpec<toy_text::NChainEnvSpec>;
using NChainEnvPool = PyEnvPool<toy_text::NChainEnvPool>;

using CliffWalkingEnvSpec = PyEnvSpec<toy_text::CliffWalkingEnvSpec>;
using CliffWalkingEnvPool = PyEnvPool<toy_text::CliffWalkingEnvPool>;

using BlackjackEnvSpec = PyEnvSpec<toy_text::BlackjackEnvSpec>;
using BlackjackEnvPool = PyEnvPool<toy_text::BlackjackEnvPool>;

PYBIND11_MODULE(toy_text_envpool, m) {
  REGISTER(m, CatchEnvSpec, CatchEnvPool)
  REGISTER(m, FrozenLakeEnvSpec, FrozenLakeEnvPool)
  REGISTER(m, TaxiEnvSpec, TaxiEnvPool)
  REGISTER(m, NChainEnvSpec, NChainEnvPool)
  REGISTER(m, CliffWalkingEnvSpec, CliffWalkingEnvPool)
  REGISTER(m, BlackjackEnvSpec, BlackjackEnvPool)
}
