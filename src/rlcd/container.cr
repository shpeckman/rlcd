# src/rlcd/container.cr
require "./checksum"
require "./format"

module RLCD
  module Container
    MAGIC      = "RLCD".to_slice
    VERSION    = 1_u8
    BIG_ENDIAN = 0x01_u8
    TAG_SIZE   = 4
    TAG        = {featurizer: "FEAT", labels: "LABL", temperature: "TEMP", policy: "POLY", optimizer: "ADAM", checkpoint: "CKPT"}

    class Writer
      getter chunks = [] of {String, IO::Memory}

      def add(tag : String, & : IO::Memory ->) : Nil
        raise ArgumentError.new("chunk tag must be #{TAG_SIZE} bytes: #{tag.inspect}") unless tag.bytesize == TAG_SIZE
        payload = IO::Memory.new
        yield payload
        @chunks << {tag, payload}
      end
    end

    struct Chunks
      getter format : IO::ByteFormat

      def initialize(@format : IO::ByteFormat, @payloads : Hash(String, Bytes))
      end

      def decode(tag : String, & : IO::Memory -> T) : T forall T
        bytes = @payloads[tag]? || raise FormatError.new("missing #{tag} chunk")
        decode_payload(tag, bytes) { |io| yield io }
      end

      def decode?(tag : String, & : IO::Memory -> T) : T? forall T
        bytes = @payloads[tag]? || return nil
        decode_payload(tag, bytes) { |io| yield io }
      end

      private def decode_payload(tag : String, bytes : Bytes, & : IO::Memory -> T) : T forall T
        io    = IO::Memory.new(bytes, writable: false)
        value = yield io
        raise FormatError.new("#{io.size - io.pos} trailing bytes in #{tag} chunk") unless io.pos == io.size
        value
      rescue IO::EOFError
        raise FormatError.new("truncated #{tag} chunk")
      end
    end

    def self.write(io : IO, format : IO::ByteFormat, & : Writer ->) : Nil
      writer = Writer.new
      yield writer
      chunks = writer.chunks
      raise ArgumentError.new("too many chunks: #{chunks.size}") if chunks.size > UInt16::MAX
      sum = Checksum.new(io)
      sum.write(MAGIC)
      sum.write_byte(VERSION)
      sum.write_byte(format == IO::ByteFormat::BigEndian ? BIG_ENDIAN : 0_u8)
      sum.write_bytes(chunks.size.to_u16, format)
      chunks.each do |(tag, payload)|
        sum.write(tag.to_slice)
        sum.write_bytes(payload.size.to_u32, format)
        sum.write(payload.to_slice)
      end
      io.write_bytes(sum.value, format)
    end

    def self.read(io : IO) : Chunks
      sum   = Checksum.new(io)
      magic = Bytes.new(MAGIC.size)
      sum.read_fully(magic)
      raise FormatError.new("not an rlcd binary model") unless magic == MAGIC
      version = sum.read_byte || raise IO::EOFError.new
      raise FormatError.new("unsupported format version #{version}") unless version == VERSION
      flags = sum.read_byte || raise IO::EOFError.new
      raise FormatError.new("unknown format flags 0x#{flags.to_s(16)}") unless flags & ~BIG_ENDIAN == 0
      format   = flags.bits_set?(BIG_ENDIAN) ? IO::ByteFormat::BigEndian : IO::ByteFormat::LittleEndian
      payloads = Hash(String, Bytes).new
      tag      = Bytes.new(TAG_SIZE)
      sum.read_bytes(UInt16, format).times do
        sum.read_fully(tag)
        name = String.new(tag)
        size = sum.read_bytes(UInt32, format)
        raise FormatError.new("duplicate #{name} chunk") if payloads.has_key?(name)
        payload = IO::Memory.new
        raise FormatError.new("truncated #{name} chunk") unless IO.copy(sum, payload, size) == size
        payloads[name] = payload.to_slice
      end
      checksum = sum.value
      raise FormatError.new("checksum mismatch") unless io.read_bytes(UInt32, format) == checksum
      Chunks.new(format, payloads)
    rescue IO::EOFError
      raise FormatError.new("truncated binary model")
    end

    def self.expect_remaining(io : IO::Memory, bytes : Int64, what : String) : Nil
      remaining = io.size - io.pos
      raise FormatError.new("#{what} holds #{remaining} bytes, expected #{bytes}") unless remaining == bytes
    end
  end
end
