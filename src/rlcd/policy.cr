# src/rlcd/policy.cr
require "json"

module RLCD
  record Forward, hidden : Array(Float64), logits : Array(Float64), probs : Array(Float64)

  class Policy
    getter input_dim    : Int32
    getter hidden_dim   : Int32
    getter option_count : Int32
    getter w1           : Mat
    getter w2           : Mat
    getter b1           : Array(Float64)
    getter b2           : Array(Float64)

    def initialize(@input_dim : Int32, @hidden_dim : Int32, @option_count : Int32, @rng : Random = Random.new)
      raise ArgumentError.new("input_dim must be positive") if @input_dim <= 0
      raise ArgumentError.new("option_count must be at least 2") if @option_count < 2
      raise ArgumentError.new("hidden_dim must be non-negative") if @hidden_dim < 0
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

    def save(path : String) : Nil
      JSON.build(File.open(path, "w")) do |json|
        json.object do
          json.field "input_dim", @input_dim
          json.field "hidden_dim", @hidden_dim
          json.field "option_count", @option_count
          json.field "w1", @w1.data.to_a
          json.field "b1", @b1
          json.field "w2", @w2.data.to_a
          json.field "b2", @b2
        end
      end
    end

    def self.load(path : String, rng : Random = Random.new) : Policy
      data         = JSON.parse(File.read(path))
      input_dim    = data["input_dim"].as_i.to_i32
      hidden_dim   = data["hidden_dim"].as_i.to_i32
      option_count = data["option_count"].as_i.to_i32
      policy       = new(input_dim, hidden_dim, option_count, rng)

      copy = ->(any : JSON::Any, slice : Slice(Float64)) do
        vals = any.as_a
        raise ArgumentError.new("saved weights length #{vals.size} does not match expected #{slice.size}") if vals.size != slice.size
        slice.size.times { |i| slice[i] = vals[i].as_f }
      end
      copy.call(data["w1"], policy.w1.data)
      copy.call(data["w2"], policy.w2.data)
      copy.call(data["b1"], Slice(Float64).new(policy.b1.to_unsafe, policy.b1.size))
      copy.call(data["b2"], Slice(Float64).new(policy.b2.to_unsafe, policy.b2.size))
      policy
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
