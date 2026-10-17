# src/rlcd.cr
require "./rlcd/*"

module Rlcd
  VERSION = {{ `shards version "#{__DIR__}/.."`.chomp.stringify }}
end
