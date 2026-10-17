# src/rlcd/mat.cr
module Rlcd
  struct Mat
    getter rows : Int32
    getter cols : Int32
    getter data : Slice(Float64)

    def initialize(@rows : Int32, @cols : Int32)
      @data = Slice(Float64).new(@rows * @cols, 0.0)
    end

    def initialize(@rows : Int32, @cols : Int32, @data : Slice(Float64))
    end

    def self.random(rows : Int32, cols : Int32, rng : Random) : Mat
      scale = Math.sqrt(2.0 / (rows + cols))
      data  = Slice(Float64).new(rows * cols) { (rng.next_float * 2.0 - 1.0) * scale }
      new(rows, cols, data)
    end

    def [](r : Int32, c : Int32) : Float64
      @data[r * @cols + c]
    end

    def []=(r : Int32, c : Int32, v : Float64)
      @data[r * @cols + c] = v
    end

    def mul_vec(x : Array(Float64)) : Array(Float64)
      result = Array(Float64).new(@rows, 0.0)
      @rows.times do |r|
        base = r * @cols
        acc  = 0.0
        @cols.times { |c| acc += @data[base + c] * x[c] }
        result[r] = acc
      end
      result
    end

    def t_mul_vec(x : Array(Float64)) : Array(Float64)
      result = Array(Float64).new(@cols, 0.0)
      @rows.times do |r|
        base = r * @cols
        v    = x[r]
        @cols.times { |c| result[c] += @data[base + c] * v }
      end
      result
    end

    def add_outer!(a : Array(Float64), b : Array(Float64), scale : Float64 = 1.0)
      @rows.times do |r|
        base = r * @cols
        av   = a[r] * scale
        @cols.times { |c| @data[base + c] += av * b[c] }
      end
    end
  end
end
