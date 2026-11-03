# src/rlcd/persistent.cr
require "json"
require "./format"

module RLCD
  module Persistent
    def save(path : String, format : Format = Format.from_path(path), precision : Precision = Precision::F64) : Nil
      raise ArgumentError.new("precision #{precision} applies to the binary format only") if format.json? && !precision.f64?
      File.open(path, "w") do |file|
        case format
        in .json?   then to_json(file)
        in .binary? then to_io(file, IO::ByteFormat::LittleEndian, precision)
        end
      end
    end

    macro included
      def self.load(path : String, rng : Random = Random.new) : self
        File.open(path) do |file|
          case Format.detect(file.peek)
          in .json?   then new(JSON::PullParser.new(file), rng)
          in .binary? then from_io(file, IO::ByteFormat::LittleEndian, rng)
          end
        end
      end
    end
  end
end
