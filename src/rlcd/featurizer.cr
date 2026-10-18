# src/rlcd/featurizer.cr
module Rlcd
  record Decision, state : String, question : String, criteria : String

  class Featurizer
    getter dim : Int32

    def initialize(@dim : Int32 = 128, @seed : UInt64 = 0xcbf29ce484222325_u64)
      raise ArgumentError.new("dim must be positive") if @dim <= 0
    end

    def features(decision : Decision) : Array(Float64)
      features(decision.state, decision.question, decision.criteria)
    end

    def features(state : String, question : String, criteria : String) : Array(Float64)
      vec = Array(Float64).new(@dim, 0.0)
      tokenize(state).each { |t| add_token(vec, t, 0_u64) }
      tokenize(question).each { |t| add_token(vec, t, 1_u64) }
      tokenize(criteria).each { |t| add_token(vec, t, 2_u64) }
      norm = Math.sqrt(vec.sum { |v| v * v })
      norm > 1e-12 ? vec.map { |v| v / norm } : vec
    end

    private def tokenize(text : String) : Array(String)
      text.downcase.split(/[^a-z0-9]+/).reject { |t| t.size < 2 }
    end

    private def add_token(vec : Array(Float64), token : String, field : UInt64) : Nil
      h    = hash_token(field, token)
      idx  = (h % @dim.to_u64).to_i
      sign = ((h >> 63) & 1_u64) == 1_u64 ? -1.0 : 1.0
      vec[idx] += sign
    end

    private def hash_token(field : UInt64, token : String) : UInt64
      h = @seed ^ (field &* 0x9e3779b97f4a7c15_u64)
      token.each_byte do |b|
        h ^= b.to_u64
        h &*= 0x100000001b3_u64
      end
      h
    end
  end
end
