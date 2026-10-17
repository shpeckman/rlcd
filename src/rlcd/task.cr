# src/rlcd/task.cr
module Rlcd
  class SyntheticTask
    getter centers : Array(Array(Float64))

    def initialize(
      @input_dim   : Int32   = 16,
      @class_count : Int32   = 4,
      @noise       : Float64 = 1.2,
      @rng         : Random  = Random.new,
    )
      @centers = Array.new(@class_count) do
        Array.new(@input_dim) { (@rng.next_float * 2.0 - 1.0) * 1.5 }
      end
    end

    def generate(count : Int32) : Array(Example)
      Array.new(count) do
        label    = @rng.rand(@class_count)
        center   = @centers[label]
        features = Array.new(@input_dim) { |i| center[i] + gaussian * @noise }
        Example.new(features, label)
      end
    end

    private def gaussian : Float64
      u1 = @rng.next_float.clamp(1e-12, 1.0)
      u2 = @rng.next_float
      Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
    end
  end
end
