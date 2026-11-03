# src/rlcd/policy.cr
require "json"
require "./container"
require "./format"
require "./mat"
require "./persistent"

module RLCD
  record Forward, hidden : Array(Float64), logits : Array(Float64), probs : Array(Float64)

  class Policy
    include Persistent

    getter input_dim    : Int32
    getter hidden_dim   : Int32
    getter option_count : Int32
    getter w1           : Mat
    getter w2           : Mat
    getter b1           : Array(Float64)
    getter b2           : Array(Float64)

    def initialize(@input_dim : Int32, @hidden_dim : Int32, @option_count : Int32, @rng : Random = Random.new)
      if message = Policy.shape_error(@input_dim, @hidden_dim, @option_count)
        raise ArgumentError.new(message)
      end
      if @hidden_dim == 0
        @w1 = Mat.new(0, @input_dim)
        @b1 = Array(Float64).new(0)
        @w2 = Mat.random(@option_count, @input_dim, @rng)
      else
        @w1 = Mat.random(@hidden_dim, @input_dim, @rng)
        @b1 = Array(Float64).new(@hidden_dim, 0.0)
        @w2 = Mat.random(@option_count, @hidden_dim, @rng)
      end
      @b2 = Array(Float64).new(@option_count, 0.0)
    end

    def self.new(pull : JSON::PullParser, rng : Random = Random.new) : Policy
      input_dim    = nil.as(Int32?)
      hidden_dim   = nil.as(Int32?)
      option_count = nil.as(Int32?)
      arrays       = {} of String => Array(Float64)
      pull.read_object do |key|
        case key
        when "input_dim"    then input_dim = Int32.new(pull)
        when "hidden_dim"   then hidden_dim = Int32.new(pull)
        when "option_count" then option_count = Int32.new(pull)
        else
          if pull.kind.begin_array?
            arrays[key] = Array(Float64).new(pull)
          else
            pull.skip
          end
        end
      end

      input   = input_dim || raise FormatError.new("policy is missing input_dim")
      hidden  = hidden_dim || raise FormatError.new("policy is missing hidden_dim")
      options = option_count || raise FormatError.new("policy is missing option_count")
      check_shape(input, hidden, options)
      layout = layout(input, hidden, options)
      layout.each do |(name, rows, cols)|
        size = (arrays[name]? || raise FormatError.new("policy is missing #{name}")).size
        raise FormatError.new("saved weights length #{size} does not match expected #{rows * cols} for #{name}") unless size == rows * cols
      end

      policy = new(input, hidden, options, rng)
      slices = policy.param_slices
      layout.each_with_index do |(name, _, _), i|
        slice = slices[i]
        arrays[name].each_with_index { |value, j| slice[j] = value }
      end
      policy
    end

    def self.from_io(io : IO, format : IO::ByteFormat = IO::ByteFormat::LittleEndian, rng : Random = Random.new) : Policy
      chunks = Container.read(io)
      chunks.decode(Container::TAG[:policy]) { |payload| decode(payload, chunks.format, rng) }
    end

    def self.decode(io : IO::Memory, format : IO::ByteFormat, rng : Random = Random.new) : Policy
      input_dim    = io.read_bytes(Int32, format)
      hidden_dim   = io.read_bytes(Int32, format)
      option_count = io.read_bytes(Int32, format)
      precision    = io.read_bytes(Precision, format)
      check_shape(input_dim, hidden_dim, option_count)
      layout = layout(input_dim, hidden_dim, option_count)
      Container.expect_remaining(io, precision.layout_bytes(layout), "policy weights")

      policy = new(input_dim, hidden_dim, option_count, rng)
      slices = policy.param_slices
      layout.each_with_index { |(_, _, cols), i| precision.decode(io, slices[i], cols, format) }
      policy
    end

    def self.shape_error(input_dim : Int32, hidden_dim : Int32, option_count : Int32) : String?
      return "input_dim must be positive" if input_dim <= 0
      return "option_count must be at least 2" if option_count < 2
      return "hidden_dim must be non-negative" if hidden_dim < 0
      "a weight tensor exceeds #{Int32::MAX} values" if layout(input_dim, hidden_dim, option_count).any? { |(_, rows, cols)| rows.to_i64 * cols > Int32::MAX }
    end

    def self.layout(input_dim : Int32, hidden_dim : Int32, option_count : Int32) : Layout
      if hidden_dim == 0
        [{"w2", option_count, input_dim}, {"b2", 1, option_count}]
      else
        [{"w1", hidden_dim, input_dim}, {"b1", 1, hidden_dim}, {"w2", option_count, hidden_dim}, {"b2", 1, option_count}]
      end
    end

    private def self.check_shape(input_dim : Int32, hidden_dim : Int32, option_count : Int32) : Nil
      if message = shape_error(input_dim, hidden_dim, option_count)
        raise FormatError.new(message)
      end
    end

    def layout : Layout
      Policy.layout(@input_dim, @hidden_dim, @option_count)
    end

    def forward(x : Array(Float64)) : Forward
      if @hidden_dim == 0
        z2     = @w2.mul_vec(x)
        logits = Array(Float64).new(@option_count) { |i| z2[i] + @b2[i] }
        return Forward.new(Array(Float64).new(0), logits, softmax(logits))
      end
      z1     = @w1.mul_vec(x)
      hidden = Array(Float64).new(@hidden_dim) { |i| Math.tanh(z1[i] + @b1[i]) }
      z2     = @w2.mul_vec(hidden)
      logits = Array(Float64).new(@option_count) { |i| z2[i] + @b2[i] }
      Forward.new(hidden, logits, softmax(logits))
    end

    def probs(x : Array(Float64)) : Array(Float64)
      forward(x).probs
    end

    def sample(x : Array(Float64)) : Int32
      categorical(forward(x).probs)
    end

    def param_slices : Array(Slice(Float64))
      if @hidden_dim == 0
        [@w2.data, Slice(Float64).new(@b2.to_unsafe, @b2.size)]
      else
        [@w1.data, Slice(Float64).new(@b1.to_unsafe, @b1.size), @w2.data, Slice(Float64).new(@b2.to_unsafe, @b2.size)]
      end
    end

    def to_json(json : JSON::Builder) : Nil
      slices = param_slices
      json.object do
        json.field "input_dim", @input_dim
        json.field "hidden_dim", @hidden_dim
        json.field "option_count", @option_count
        layout.each_with_index do |(name, _, _), i|
          json.field(name) { json.array { slices[i].each { |value| json.number(value) } } }
        end
      end
    end

    def to_io(io : IO, format : IO::ByteFormat = IO::ByteFormat::LittleEndian, precision : Precision = Precision::F64) : Nil
      Container.write(io, format) do |writer|
        writer.add(Container::TAG[:policy]) { |payload| encode(payload, format, precision) }
      end
    end

    def encode(io : IO, format : IO::ByteFormat, precision : Precision) : Nil
      io.write_bytes(@input_dim, format)
      io.write_bytes(@hidden_dim, format)
      io.write_bytes(@option_count, format)
      io.write_bytes(precision, format)
      slices = param_slices
      layout.each_with_index { |(_, _, cols), i| precision.encode(io, slices[i], cols, format) }
    end

    def softmax(logits : Array(Float64)) : Array(Float64)
      max  = logits.max
      exps = logits.map { |v| Math.exp(v - max) }
      sum  = exps.sum
      exps.map { |v| v / sum }
    end

    def categorical(probs : Array(Float64)) : Int32
      r   = @rng.next_float
      acc = 0.0
      probs.each_with_index do |p, i|
        acc += p
        return i if r < acc
      end
      probs.size - 1
    end
  end
end
