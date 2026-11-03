# src/rlcd/optimizer.cr
require "json"
require "./container"
require "./format"

module RLCD
  class Adam
    getter policy : Policy
    getter lr     : Float64
    getter beta1  : Float64
    getter beta2  : Float64
    getter eps    : Float64
    getter t      : Int32
    getter m      : Array(Slice(Float64))
    getter v      : Array(Slice(Float64))

    @params : Array(Slice(Float64))

    def initialize(
      @policy : Policy,
      @lr     : Float64 = 0.005,
      @beta1  : Float64 = 0.9,
      @beta2  : Float64 = 0.999,
      @eps    : Float64 = 1e-8,
      @t      : Int32   = 0,
    )
      raise ArgumentError.new("t must be non-negative") if @t < 0
      @params = @policy.param_slices
      @m      = @params.map { |p| Slice(Float64).new(p.size, 0.0) }
      @v      = @params.map { |p| Slice(Float64).new(p.size, 0.0) }
    end

    def self.new(policy : Policy, pull : JSON::PullParser) : Adam
      t       = nil.as(Int32?)
      lr      = nil.as(Float64?)
      beta1   = nil.as(Float64?)
      beta2   = nil.as(Float64?)
      eps     = nil.as(Float64?)
      moments = {} of String => Array(Array(Float64))
      pull.read_object do |key|
        case key
        when "t"      then t = Int32.new(pull)
        when "lr"     then lr = Float64.new(pull)
        when "beta1"  then beta1 = Float64.new(pull)
        when "beta2"  then beta2 = Float64.new(pull)
        when "eps"    then eps = Float64.new(pull)
        when "m", "v" then moments[key] = Array(Array(Float64)).new(pull)
        else               pull.skip
        end
      end

      adam = new(
        policy,
        lr || raise(FormatError.new("optimizer is missing lr")),
        beta1 || raise(FormatError.new("optimizer is missing beta1")),
        beta2 || raise(FormatError.new("optimizer is missing beta2")),
        eps || raise(FormatError.new("optimizer is missing eps")),
        t || raise(FormatError.new("optimizer is missing t")),
      )
      {"m" => adam.m, "v" => adam.v}.each do |name, targets|
        saved = moments[name]? || raise FormatError.new("optimizer is missing #{name}")
        raise FormatError.new("optimizer #{name} has #{saved.size} tensors, expected #{targets.size}") unless saved.size == targets.size
        targets.each_with_index do |target, i|
          values = saved[i]
          raise FormatError.new("optimizer #{name}[#{i}] length #{values.size} does not match expected #{target.size}") unless values.size == target.size
          values.each_with_index { |value, j| target[j] = value }
        end
      end
      adam
    end

    def self.decode(policy : Policy, io : IO::Memory, format : IO::ByteFormat) : Adam
      t         = io.read_bytes(Int32, format)
      lr        = io.read_bytes(Float64, format)
      beta1     = io.read_bytes(Float64, format)
      beta2     = io.read_bytes(Float64, format)
      eps       = io.read_bytes(Float64, format)
      precision = io.read_bytes(Precision, format)
      raise FormatError.new("optimizer step count must be non-negative") if t < 0
      layout = policy.layout
      Container.expect_remaining(io, precision.layout_bytes(layout) * 2, "optimizer moments")

      adam = new(policy, lr, beta1, beta2, eps, t)
      {adam.m, adam.v}.each do |moments|
        layout.each_with_index { |(_, _, cols), i| precision.decode(io, moments[i], cols, format) }
      end
      adam
    end

    def lr=(value : Float64) : Nil
      @lr = value
    end

    def step!(grads : Array(Slice(Float64))) : Nil
      @t += 1
      bc1 = 1.0 - @beta1 ** @t
      bc2 = 1.0 - @beta2 ** @t
      @params.each_with_index do |p, i|
        g  = grads[i]
        mi = @m[i]
        vi = @v[i]
        p.size.times do |j|
          gj = g[j]
          mi[j] = @beta1 * mi[j] + (1.0 - @beta1) * gj
          vi[j] = @beta2 * vi[j] + (1.0 - @beta2) * gj * gj
          p[j] -= @lr * (mi[j] / bc1) / (Math.sqrt(vi[j] / bc2) + @eps)
        end
      end
    end

    def to_json(json : JSON::Builder) : Nil
      json.object do
        json.field "t", @t
        json.field "lr", @lr
        json.field "beta1", @beta1
        json.field "beta2", @beta2
        json.field "eps", @eps
        {"m" => @m, "v" => @v}.each do |name, moments|
          json.field(name) do
            json.array { moments.each { |slice| json.array { slice.each { |value| json.number(value) } } } }
          end
        end
      end
    end

    def encode(io : IO, format : IO::ByteFormat, precision : Precision) : Nil
      io.write_bytes(@t, format)
      io.write_bytes(@lr, format)
      io.write_bytes(@beta1, format)
      io.write_bytes(@beta2, format)
      io.write_bytes(@eps, format)
      moment_precision = precision.moment_precision
      io.write_bytes(moment_precision, format)
      layout = @policy.layout
      {@m, @v}.each do |moments|
        layout.each_with_index { |(_, _, cols), i| moment_precision.encode(io, moments[i], cols, format) }
      end
    end
  end
end
