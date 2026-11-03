# src/rlcd/checkpoint.cr
require "json"
require "./format"

module RLCD
  record Checkpoint, epoch : Int32, epochs : Int32, seed : UInt64? do
    def self.new(pull : JSON::PullParser) : Checkpoint
      epoch  = nil.as(Int32?)
      epochs = nil.as(Int32?)
      seed   = nil.as(UInt64?)
      pull.read_object do |key|
        case key
        when "epoch"  then epoch = Int32.new(pull)
        when "epochs" then epochs = Int32.new(pull)
        when "seed"   then seed = pull.read_null_or { UInt64.new(pull) }
        else               pull.skip
        end
      end
      checked(
        epoch || raise(FormatError.new("checkpoint is missing epoch")),
        epochs || raise(FormatError.new("checkpoint is missing epochs")),
        seed,
      )
    end

    def self.decode(io : IO::Memory, format : IO::ByteFormat) : Checkpoint
      epoch  = io.read_bytes(Int32, format)
      epochs = io.read_bytes(Int32, format)
      seeded = io.read_byte || raise IO::EOFError.new
      seed   = io.read_bytes(UInt64, format)
      raise FormatError.new("checkpoint seed flag must be 0 or 1, got #{seeded}") if seeded > 1
      checked(epoch, epochs, seeded == 1 ? seed : nil)
    end

    private def self.checked(epoch : Int32, epochs : Int32, seed : UInt64?) : Checkpoint
      raise FormatError.new("checkpoint epoch #{epoch} is outside 0..#{epochs}") unless 0 <= epoch <= epochs
      new(epoch, epochs, seed)
    end

    def finished? : Bool
      epoch >= epochs
    end

    def to_json(json : JSON::Builder) : Nil
      json.object do
        json.field "epoch", epoch
        json.field "epochs", epochs
        json.field "seed", seed
      end
    end

    def encode(io : IO, format : IO::ByteFormat) : Nil
      io.write_bytes(epoch, format)
      io.write_bytes(epochs, format)
      io.write_byte(seed ? 1_u8 : 0_u8)
      io.write_bytes(seed || 0_u64, format)
    end
  end
end
