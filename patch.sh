#!/bin/bash

set -e

git submodule foreach git reset --hard
cat llama-ext.cpp >> llama.cpp/llama.cpp
cat llama-ext.h >> llama.cpp/llama.h
