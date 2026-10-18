# src/rlcd.cr
require "./rlcd/*"

module RLCD
  VERSION = {{ `shards version "#{__DIR__}/.."`.chomp.stringify }}
end
