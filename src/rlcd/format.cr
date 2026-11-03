# src/rlcd/format.cr
module RLCD
  class FormatError < ArgumentError
  end

  alias Layout = Array({String, Int32, Int32})

  enum Format
    Json
    Binary

    def self.from_path(path : String) : Format
      File.extname(path).downcase == ".json" ? Json : Binary
    end

    def self.detect(bytes : Bytes?) : Format
      head = bytes || raise FormatError.new("cannot detect model format: input is not peekable")
      return Binary if head.size >= Container::MAGIC.size && head[0, Container::MAGIC.size] == Container::MAGIC
      first = head.find { |byte| !byte.unsafe_chr.ascii_whitespace? }
      return Json if first == '{'.ord
      raise FormatError.new(head.empty? ? "empty model input" : "unrecognized model format")
    end
  end

  enum Precision : UInt8
    F64
    F32
    I8

    def self.from_io(io : IO, format : IO::ByteFormat) : Precision
      byte = io.read_byte || raise IO::EOFError.new
      from_value?(byte) || raise FormatError.new("unknown precision #{byte}")
    end

    def to_io(io : IO, format : IO::ByteFormat) : Nil
      io.write_byte(value)
    end

    def moment_precision : Precision
      i8? ? F32 : self
    end

    def tensor_bytes(rows : Int32, cols : Int32) : Int64
      case self
      in .f64? then rows.to_i64 * cols * 8
      in .f32? then rows.to_i64 * cols * 4
      in .i8?  then rows.to_i64 * (cols.to_i64 + 4)
      end
    end

    def layout_bytes(layout : Layout) : Int64
      layout.sum(0_i64) { |(_, rows, cols)| tensor_bytes(rows, cols) }
    end

    def encode(io : IO, data : Slice(Float64), cols : Int32, format : IO::ByteFormat) : Nil
      case self
      in .f64? then Precision.write_values(io, data, format)
      in .f32? then Precision.write_values(io, Slice(Float32).new(data.size) { |i| data[i].to_f32 }, format)
      in .i8?
        quantized = Slice(Int8).new(cols)
        (data.size // cols).times do |r|
          row   = data[r * cols, cols]
          scale = (row.max_of(&.abs) / 127.0).to_f32
          io.write_bytes(scale, format)
          cols.times { |i| quantized[i] = scale.zero? ? 0_i8 : (row[i] / scale).round.clamp(-127.0, 127.0).to_i8 }
          io.write(quantized.to_unsafe_bytes)
        end
      end
    end

    def decode(io : IO, data : Slice(Float64), cols : Int32, format : IO::ByteFormat) : Nil
      case self
      in .f64? then Precision.read_values(io, data, format)
      in .f32?
        narrow = Slice(Float32).new(data.size)
        Precision.read_values(io, narrow, format)
        data.size.times { |i| data[i] = narrow[i].to_f64 }
      in .i8?
        quantized = Slice(Int8).new(cols)
        (data.size // cols).times do |r|
          scale = io.read_bytes(Float32, format).to_f64
          io.read_fully(quantized.to_unsafe_bytes)
          base = r * cols
          cols.times { |i| data[base + i] = quantized[i].to_f64 * scale }
        end
      end
    end

    def self.write_values(io : IO, values : Slice(T), format : IO::ByteFormat) : Nil forall T
      if format == IO::ByteFormat::SystemEndian
        io.write(values.to_unsafe_bytes)
      else
        values.each { |value| io.write_bytes(value, format) }
      end
    end

    def self.read_values(io : IO, values : Slice(T), format : IO::ByteFormat) : Nil forall T
      if format == IO::ByteFormat::SystemEndian
        io.read_fully(values.to_unsafe_bytes)
      else
        values.size.times { |i| values[i] = io.read_bytes(T, format) }
      end
    end
  end
end
